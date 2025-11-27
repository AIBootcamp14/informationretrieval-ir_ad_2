from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.settings import LLMConfig
from llm.generators import safe_chat_completion


RERANK_SYSTEM_PROMPT = """
당신은 검색 결과 문서들을 질의에 따라 재정렬하는 한국어 LLM 기반 reranker입니다.

[역할]
- 사용자의 질의(query)와 상위 검색 문서 목록(documents)을 입력으로 받아,
  "질의와 얼마나 관련이 높은지" 기준으로 문서를 재정렬합니다.
- 결과는 반드시 JSON 형식으로만 출력해야 합니다.

[입력 형식]
- query: 사용자의 최종 질문 텍스트 (standalone query)
- documents: 다음 필드를 가진 객체 리스트
  - docid: 문서의 고유 ID (문자열)
  - content: 문서 내용 (짧은 단락)

[출력 형식]
- 아래 JSON 형식만 출력하세요. JSON 밖에 다른 텍스트를 쓰지 마세요.

{
  "reranked_docids": ["docid1", "docid3", "docid2"]
}

[세부 규칙]
1. reranked_docids에는 입력으로 주어진 docid들만 포함해야 합니다.
2. 가장 관련도가 높은 문서를 리스트의 맨 앞에 두고, 점점 덜 관련 있는 순서로 나열합니다.
3. 문서가 너무 비슷하거나 모두 애매하더라도, 질의와의 관련도를 기준으로 상대적인 순위를 정하세요.
4. 입력으로 주어진 docid의 개수가 N개라면, reranked_docids에도 정확히 N개의 docid를 모두 포함하세요.
5. JSON 외에 불필요한 설명, 자연어 문장은 절대 출력하지 마세요.
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    LLM 응답에서 첫 번째 JSON 객체 부분만 추출해 파싱한다.
    실패하면 None을 반환한다.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def llm_rerank_hits(
    query_text: str,
    search_result: Dict[str, Any],
    client: OpenAI,
    llm_config: LLMConfig,
    topn: Optional[int] = None,
) -> Dict[str, Any]:
    """
    ES 검색 결과(hits)를 LLM으로 재정렬한다.

    - query_text: standalone query
    - search_result: Elasticsearch search API와 동일한 구조의 dict
    - client: OpenAI 호환 LLM 클라이언트 (base_url로 OpenAI / Upstage 모두 지원)
    - llm_config: LLM 설정 (모델명 등)
    - topn: 상위 몇 개만 사용해서 재정렬할지 (None이면 전체)
    """
    hits = search_result.get("hits", {}).get("hits", [])
    if not hits:
        return search_result

    # 필요시 상위 topn까지만 rerank 대상으로 사용
    if topn is not None and len(hits) > topn:
        candidate_hits = hits[:topn]
    else:
        candidate_hits = hits

    # docid -> hit 매핑 및 LLM에 넘길 payload 구성
    docid_to_hit: Dict[str, Dict[str, Any]] = {}
    documents_payload: List[Dict[str, str]] = []

    for hit in candidate_hits:
        source = hit.get("_source", {}) or {}
        docid = source.get("docid")
        content = source.get("content", "")

        # docid가 없으면 LLM rerank 대상에서 제외 (안전)
        if not docid:
            continue

        # content가 너무 길면 잘라서 보낸다 (토큰 절약)
        truncated_content = content
        if len(truncated_content) > 1000:
            truncated_content = truncated_content[:1000]

        docid_to_hit[docid] = hit
        documents_payload.append(
            {
                "docid": docid,
                "content": truncated_content,
            }
        )

    # 유효한 문서가 1개 이하면 굳이 rerank할 필요 없음
    if len(documents_payload) <= 1:
        return search_result

    user_payload = {
        "query": query_text,
        "documents": documents_payload,
    }

    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    result = safe_chat_completion(
        client=client,
        model=llm_config.model,
        messages=messages,
        temperature=0,
        seed=1,
        timeout=40,
        max_retries=3,
    )

    if result is None:
        # LLM 호출 실패 시 원래 결과를 그대로 반환
        return search_result

    answer_text = result.choices[0].message.content or ""
    parsed = _extract_json(answer_text)
    if not parsed:
        return search_result

    reranked_ids = parsed.get("reranked_docids")
    if not isinstance(reranked_ids, list) or not reranked_ids:
        return search_result

    # LLM이 준 순서대로 hit를 재배열
    new_hits: List[Dict[str, Any]] = []
    used_docids = set()
    for docid in reranked_ids:
        if not isinstance(docid, str):
            continue
        hit = docid_to_hit.get(docid)
        if hit is not None:
            new_hits.append(hit)
            used_docids.add(docid)

    # 혹시 누락된 docid가 있다면 원래 순서를 유지한 채 뒤에 붙인다.
    for docid, hit in docid_to_hit.items():
        if docid not in used_docids:
            new_hits.append(hit)

    # candidate 외의 나머지 hits가 있었다면, 원래 순서를 유지한 채 뒤에 그대로 붙인다.
    remaining_hits = [h for h in hits if h not in candidate_hits]
    new_hits.extend(remaining_hits)

    # search_result 구조는 유지하되 hits만 교체
    return {
        "hits": {
            "hits": new_hits,
        }
    }

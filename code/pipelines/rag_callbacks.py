"""LangGraph 콜백 구성 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from elasticsearch import Elasticsearch

from config.settings import Settings
from llm.embedding import EmbeddingService
from llm.generators import (
    build_standalone_query,
    create_llm_client,
    generate_final_answer,
)
from llm.reranker import llm_rerank_hits
from pipelines.langgraph_pipeline import RagGraphCallbacks
from retrieval.non_science import is_science_query
from retrieval.retriever import hybrid_retrieve


# RagDependencies는 LangGraph 콜백이 의존하는 자원을 묶어 둔 데이터 클래스다.
@dataclass
class RagDependencies:
    settings: Settings
    es_client: Elasticsearch
    embedder: EmbeddingService


# build_callbacks는 RagGraphCallbacks 인스턴스를 생성해 반환한다.
def build_callbacks(deps: RagDependencies) -> RagGraphCallbacks:
    # 메인 LLM 클라이언트 (질의 생성 + 최종 답변)
    main_llm_client = create_llm_client(deps.settings.llm)

    # rerank 전용 LLM 클라이언트 (설정이 없으면 메인 LLM을 재사용)
    if deps.settings.rerank_llm is not None:
        rerank_llm_client = create_llm_client(deps.settings.rerank_llm)
        rerank_llm_config = deps.settings.rerank_llm
    else:
        # fallback: rerank도 메인 LLM 사용
        rerank_llm_client = main_llm_client
        rerank_llm_config = deps.settings.llm

    # classify_query는 규칙 기반 비과학 판별기를 호출한다.
    def classify_query(messages: List[Dict]):
        return is_science_query(messages)

    # build_query는 standalone query 생성을 담당한다. (메인 LLM 사용)
    def build_query(messages: List[Dict]):
        return build_standalone_query(messages, main_llm_client, deps.settings.llm)

    # retrieve는 하이브리드 검색을 호출한 뒤, 옵션에 따라 LLM rerank를 적용한다.
    def retrieve(query_text: str, kwargs: Dict):
        size = kwargs.get("size", 3)
        alpha = kwargs.get("alpha", 0.5)
        use_llm_rerank = kwargs.get("use_llm_rerank", False)
        rerank_topn = kwargs.get("rerank_topn")  # 없으면 size를 기본으로 사용

        # 1) hybrid 검색 (BM25 + dense)
        search_result = hybrid_retrieve(
            client=deps.es_client,
            index_name=deps.settings.es.index_name,
            query_text=query_text,
            size=size,
            embedder=deps.embedder,
            alpha=alpha,
        )

        # 2) 필요시 LLM 기반 rerank (가능하면 OpenAI rerank 전용 모델 사용)
        if use_llm_rerank:
            topn = rerank_topn or size
            search_result = llm_rerank_hits(
                query_text=query_text,
                search_result=search_result,
                client=rerank_llm_client,
                llm_config=rerank_llm_config,
                topn=topn,
            )

        return search_result

    # generate_answer는 검색 결과를 바탕으로 최종 답변을 생성한다. (메인 LLM 사용)
    def generate_answer(messages: List[Dict], docs: List[str]):
        return generate_final_answer(messages, docs, main_llm_client, deps.settings.llm)

    return RagGraphCallbacks(
        classify_query=classify_query,
        build_query=build_query,
        retrieve=retrieve,
        generate_answer=generate_answer,
    )

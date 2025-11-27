"""환경 설정 로더 모듈."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


# Elasticsearch 접속 정보를 표현하는 데이터 클래스.
@dataclass
class ElasticsearchConfig:
    host: str
    username: Optional[str]
    password: Optional[str]
    ca_cert: Optional[str]
    index_name: str


# LLM 호출에 필요한 설정을 표현하는 데이터 클래스.
@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str


# 전체 실행에 필요한 설정을 하나로 모은 데이터 클래스.
@dataclass
class Settings:
    es: ElasticsearchConfig
    llm: LLMConfig                     # 메인 LLM (질의 생성 + 최종 답변)
    rerank_llm: Optional[LLMConfig] = None  # rerank 전용 LLM (예: OpenAI)


# load_settings는 .env에서 환경 변수를 읽고 Settings 객체를 반환한다.
def load_settings(default_model: str = "solar-pro2") -> Settings:
    load_dotenv()

    # ============================
    # 1) Elasticsearch 설정
    # ============================
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    es_username = os.getenv("ES_USERNAME")
    es_password = os.getenv("ES_PASSWORD")
    es_ca_cert = os.getenv("ES_CA_CERT")
    es_index_name = os.getenv("ES_INDEX", "test")

    es_config = ElasticsearchConfig(
        host=es_host,
        username=es_username,
        password=es_password,
        ca_cert=es_ca_cert,
        index_name=es_index_name,
    )

    # ============================
    # 2) 메인 LLM 설정 (Upstage / OpenAI 공용)
    # ============================
    # 우선순위: OPENAI_API_KEY > SOLAR_API_KEY
    openai_api_key = os.getenv("OPENAI_API_KEY")
    solar_api_key = os.getenv("SOLAR_API_KEY")
    llm_api_key = openai_api_key or solar_api_key
    if llm_api_key is None:
        raise RuntimeError("OPENAI_API_KEY 또는 SOLAR_API_KEY 중 하나는 반드시 설정해야 합니다.")

    # 기본으로는 Upstage API를 사용하되, 필요하면 LLM_BASE_URL로 override 가능
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.upstage.ai/v1")
    llm_model = os.getenv("LLM_MODEL", default_model)

    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
    )

    # ============================
    # 3) rerank 전용 LLM 설정 (OpenAI 전용, 선택 사항)
    # ============================
    # - RERANK_OPENAI_API_KEY 가 설정되어 있으면 rerank는 항상 OpenAI로 보냄.
    # - 없으면 rerank도 메인 LLM (llm_config)를 그대로 사용.
    rerank_openai_api_key = os.getenv("RERANK_OPENAI_API_KEY")
    rerank_llm_config: Optional[LLMConfig] = None

    if rerank_openai_api_key:
        # 기본값은 가볍게 쓸 수 있는 OpenAI 모델로 설정 (원하면 .env에서 override)
        rerank_model = os.getenv("RERANK_LLM_MODEL", "gpt-4.1-mini")
        rerank_base_url = os.getenv("RERANK_LLM_BASE_URL", "https://api.openai.com/v1")

        rerank_llm_config = LLMConfig(
            api_key=rerank_openai_api_key,
            model=rerank_model,
            base_url=rerank_base_url,
        )

    return Settings(es=es_config, llm=llm_config, rerank_llm=rerank_llm_config)

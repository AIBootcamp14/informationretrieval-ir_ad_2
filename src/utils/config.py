# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """환경 변수 및 설정 관리"""
    
    # API Keys
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

    # API Key 검증
    @classmethod
    def validate_api_keys(cls):
        """필수 API 키 존재 여부 확인"""
        if not cls.UPSTAGE_API_KEY:
            raise ValueError("UPSTAGE_API_KEY 설정되지 않았습니다.")
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Logs
    LOGS_DIR = BASE_DIR / "logs"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM Models
    LLM_MODEL = "solar-pro2"  # Financial Analyst와 Report Generator가 사용할 기본 모델
    LLM_TEMPERATURE = 0  # 기본 temperature (0 = 결정적)

    # Retriever
    SCRIPT_PATH = os.path.join(BASE_DIR, 'bash_script', 'run_elasticsearch.sh')
    INDEX = 'test'
    EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    SETTING = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }

    MAPPINGS = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "l2_norm"
            }
        }
    }

    # DATA PATH
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'documents.jsonl')
    EVAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'eval.jsonl')
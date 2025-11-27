# src/agents/request_analyst.py
"""
Request Analyst Module

사용자의 요청이 경제/금융 관련인지 판별하는 분류기입니다.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.llm.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScienceGate(BaseModel):
    """과학상식 관련 질문 여부 판단"""
    label: Literal["science", "not_science"] = Field(
        description="질문 분류: 'science' (과학상식 관련), 'not_science'"
    )
def science_cls(state, llm=None):
    logger.info("=" * 10 + " science query classification start!! " + "=" * 10)
    question = state['standalone_query']
    logger.info(f"판단할 과학질문 : {question}")

    if llm is None:
        llm_manager = get_llm_manager()
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)

    llm_manager = get_llm_manager()
    prompt = llm_manager.get_prompt('science_cls')

    chain = prompt | llm.with_structured_output(ScienceGate)
    result = chain.invoke({"input": question})
    logger.info(f"판단 결과 : {result.label}")

    return result.label

if __name__ == "__main__":
    question1 = {'standalone_query':"물체가 떨어질 때 속도가 빨라지는 이유는 뭐야?"}    
    question2 = {'standalone_query':"지진은 왜 발생해?"}
    question3 = {'standalone_query':"삼성전자 주가는 왜 떨어져?"}
    question4 = {'standalone_query':"전기세가 왜 비싸졌어?"}
    question5 = {'standalone_query':"나 요즘 힘들어"}

    science_cls(question1)
    science_cls(question2)
    science_cls(question3)
    science_cls(question4)
    science_cls(question5)
    
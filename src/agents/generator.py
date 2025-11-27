# src/agents/query_cleaner.py
"""
generator Module

사용자의 질문과 retriever된 문서들을 가지고 적절한 답변을 생성해내는 모듈입니다. 
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from src.llm.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnswerGate(BaseModel):
    """답변"""
    answer: str = Field(description="사용자의 질문과 검색된 문서를 이용하여 적절한 답변 생성")

def generator(state: Dict[str, Any], llm=None, reference = None) -> Dict[str, str]:
    logger.info("=" * 10 + " 답변 생성 시작 " + "=" * 10)

    question = state['standalone_query']
    if reference is None:
        reference = [i['content'] for i in state['references']]
    reference_text = "\n\n".join(reference)
    # LLM 가져오기
    if llm is None:
        llm_manager = get_llm_manager()
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        logger.info(f"기본 LLM 모델 사용: {Config.LLM_MODEL}")

    # 프롬프트 가져오기
    llm_manager = get_llm_manager()
    prompt = llm_manager.get_prompt('generator')
    chain = prompt | llm.with_structured_output(AnswerGate)
    result = chain.invoke({"input": question, "reference": reference_text})

    logger.info(f"Generator Answer: {result.answer[:30]} ...")
    return result.answer
    

if __name__ == "__main__" :
    question1 = {'standalone_query': "파이썬이 뭐야?", 'references': [{'content' : "파이썬 변수 이름은 대소문자를 구분합니다. 파이썬은 대소문자를 구분하는 언어이기 때문에 변수 이름을 정확하게 입력해야 합니다. 예를 들어, 'myVariable'과 'myvariable'은 서로 다른 변수로 인식됩니다. 이러한 특성은 파이썬의 식별자 규칙에 따라 동작합니다. 따라서, 변수를 사용할 때는 대소문자를 정확하게 구분해야 합니다."},
                                                               {'content' : '파이썬에서 4*1**4의 출력은 4입니다. 파이썬은 강력한 프로그래밍 언어로, 수학적인 연산도 쉽게 처리할 수 있습니다. 이 연산은 4를 1에 4번 곱한 결과를 나타냅니다. 파이썬에서 ** 연산자는 거듭제곱을 의미하며, 1의 4제곱은 1을 4번 곱한 것과 같습니다. 따라서 4*1**4는 4를 의미합니다. 파이썬은 다양한 연산을 지원하며, 이를 통해 다양한 계산을 수행할 수 있습니다.'},
                                                               {'content' : 'PAL(Programmable Array Logic)은 프로그래밍 가능한 AND 어레이와 고정 OR 어레이로 구성된 PLD(Programmable Logic Device)의 일종입니다. 이는 복잡한 논리 기능을 구현하기 위해 디지털 회로 설계에 일반적으로 사용됩니다. PAL은 광범위한 논리 연산을 수행할 수 있는 능력으로 알려져 있어 다재다능하고 유연합니다. 프로그래머블 AND 어레이는 사용자가 입력과 AND 게이트 간의 연결을 프로그래밍하여 원하는 논리 기능을 정의할 수 있도록 합니다. 고정 OR 어레이는 이후 고정 연결을 사용하여 AND 게이트의 출력을 결합하여 PAL의 최종 출력을 산출합니다. 이러한 프로그래머블 어레이와 고정 어레이의 조합은 PAL의 이름을 부여하고 다른 유형의 PLD와 구별합니다. PAL은 데이터 처리, 제어 시스템 및 통신을 포함한 다양한 응용 분야에서 널리 사용됩니다. 이들은 복잡한 논리 기능을 콤팩트하고 효율적인 방식으로 구현하기 위한 비용 효율적인 솔루션을 제공합니다.'}]}
    result = generator(question1)
    print(result)
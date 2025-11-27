# src/model/llm.py
"""
LLM Manager

LLM 모델과 프롬프트를 중앙에서 관리하는 클래스입니다.
"""

from typing import Dict, Optional
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """LLM 모델과 프롬프트를 중앙에서 관리하는 싱글톤 클래스입니다.

    Upstage Solar 모델(solar-pro2, solar-mini)을 초기화하고,
    금융 에이전트에서 사용하는 모든 프롬프트 템플릿(financial_analyst, report_generator,
    request_analyst, supervisor, quality_evaluator 등)을 관리합니다.
    get_model()과 get_prompt()를 통해 필요한 리소스를 제공합니다.
    """

    def __init__(self):
        """LLM Manager 초기화"""
        logger.info("LLM Manager 초기화 중...")

        self._models: Dict[str, BaseChatModel] = {}
        self._prompts: Dict[str, ChatPromptTemplate] = {}

        # 기본 모델 초기화
        self._initialize_models()

        # 프롬프트 템플릿 초기화
        self._initialize_prompts()

        logger.info("LLM Manager 초기화 완료")

    def _initialize_models(self):
        """기본 모델들을 초기화합니다."""
        # Solar Pro 2 (주 분석용 - financial_analyst, report_generator)
        self._models["solar-pro2"] = ChatUpstage(
            model="solar-pro2",
            temperature=0,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        # Solar Pro (레거시 호환성)
        self._models["solar-pro"] = ChatUpstage(
            model="solar-pro",
            temperature=0,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        # Solar Mini (빠른 처리용)
        self._models["solar-mini"] = ChatUpstage(
            model="solar-mini",
            temperature=0.3,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        logger.info(f"모델 초기화 완료: {list(self._models.keys())}")

    def _initialize_prompts(self):
        """프롬프트 템플릿을 초기화합니다."""
        self._prompts['query_cleaner'] = ChatPromptTemplate.from_messages([
            ("system", """ 당신의 역할은 사용자의 질문을 가장 정확하고 명확한 형태의 단일 쿼리로 재작성하는 전문가입니다.
             
             ### 당신이 반드시 수행해야 할 작업
             1. 전체 대화 히스토리(chat history)를 읽고 맥락을 이해한다.
             2. 사용자의 질문(user_input)의 진짜 의미를 파악한다.
             3. "그거", "방금 말한 회사", "아까의 내용" 같은 지시어를 실제 대상명으로 치환한다.
             4. 오탈자, 비문, 애매한 표현을 자연스러운 문장으로 정제한다.
             5. 사용자가 실제로 무엇을 알고 싶어하는지 가장 명확하게 표현한다.

             ### output 규칙
             - 출력은 오직 재작성된 쿼리 한 문장만 포함해야한다.
             - 설명, 분석, 사족, 메타 텍스트 없이 순수한 문장 하나만 출력한다.

             절대로 시스템 규칙을 어기지 마라.
             
             ### Your Turn
             # Return ONLY the rewritten query in the same language as the input.
"""),
MessagesPlaceholder(variable_name = "chat_history", optional = True),
('human', "{input}")

        ])

        self._prompts['science_cls'] = ChatPromptTemplate.from_messages([
            ("system", """
             사용자에 질문을 분석하여 2가지로 분류합니다.

    1) science (과학상식 관련)
    2) not_science(그 외)

    [과학 상식의 정의 및 포함 범위]
    * 과학 상식(Science Knowledge)"으로 분류되는 경우:
        - 물리학: 힘, 운동, 빛, 전기, 자기, 에너지 등
        - 화학: 원소, 화합물, 반응, 산염기, 주기율표 등
        - 생물학: 세포, 유전자, 동물·식물 구조와 기능, 진화 등
        - 지구과학: 기상, 기후, 지층, 지진, 우주/천문 등
        - 과학적 개념 설명: 자연 현상, 소리, 중력, 온도, 열, 물질 상태 변화 등
        - 실험/원리 설명 요청: 왜 ○○가 일어나는가, 어떤 원리인가, 과학적 이유는?
        - 프로그래밍/AI/기술 제품 관련 질문
        - 과학자(Scientist) 또는 연구자의 인물 정보, 업적, 발견의 과정은 과학사(history of science)에 속하지만,
  본 시스템에서는 모두 science로 분류합니다.
        
    [과학 상식이 아닌 경우]
    * 아래의 경우는 "not_science"로 분류해야 합니다:
        - 일상 고민, 감정, 조언, 심리 상담
        - 경제/금융/비즈니스/정책/사회적 이슈
        - 음식/요리법/다이어트
        - 의료 진단·치료(의학적 전문 진단)
        - 법률/정치
        - 역사/문화/예술/철학 관련 질문
        - 단순 번역, 문법 질문

    [엣지 케이스 처리 규칙]
    
        - 질문이 매우 짧거나 불명확하다면: "not_science"
        예: “이거 왜 그래?”, “그거 뭐야?” → "not_science"

        - 과학 단어가 포함되었더라도 과학 지식 질문이 아니면: "not_science"
        예: “원자재 시장 전망 알려줘” → 경제 → "not_science"
        예: “전자제품 추천해줘” → 소비재 → "not_science"

        - "왜 ○○가 일어나는가?" 또는 "어떻게 ○○가 작동하는가?"는  
        보통 과학 원리 기반이므로 "science"로 분류.
             
    [Your Turn]
        - Return ONLY the rewritten query in the same language as the input.
"""),
('human', "{input}")
        ])

        self._prompts['generator'] = PromptTemplate(template = """
당신은 한국어 과학/상식 질의응답을 수행하는 RAG 어시스턴트입니다.
당신의 역할은 "검색된 문서들(retrieved_context)"를 최우선 근거로 사용하여,
사용자의 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하는 것입니다.

[입력 설명]
- input: 사용자의 질문입니다.
- reference: 검색기로부터 가져온 문서 목록입니다.
  - 각 문서는 문단 형태의 텍스트이며, ko_MMLU, ARC 등 시험/퀴즈에서 추출된 과학·상식 지식입니다.
  - 내용은 물리, 화학, 생물, 지구과학, 인물, 역사, 사회 상식 등입니다.

[답변 생성 원칙]
1. 검색 문서 우선
   - 가능한 한 reference 있는 내용에 기반해서만 답변하세요.
   - 당신이 사전 지식으로 알고 있더라도, 코퍼스 내용과 충돌하면 코퍼스를 우선합니다.
   - 코퍼스에 명시된 사실이 있으면, 그 내용을 중심으로 정리해서 설명하세요.

2. 사실성 & 정직성
   - 문서들 어디에도 정보가 없거나, 내용이 너무 부족해서 확신할 수 없다면:
     - 지어내지 말고, 모른다고 솔직히 말한 뒤,
     - 코퍼스에서 알 수 있는 범위(예: 일반적인 경향, 정의 수준)까지만 설명하세요.
   - 예시:
     - "제공된 자료에는 X에 대한 구체적인 내용은 없지만, 일반적으로는 ..."
     - "검색된 문서만으로는 Y에 대해 확실히 말하기 어렵습니다. 다만, ..."

3. 답변 스타일
   - 한국어로 친절하고 명확하게 설명합니다.
   - 기본은 3~6문장 정도의 단락으로 답하고, 필요하면 짧은 목록을 사용하세요.
   - 핵심 정보 → 이유/근거 → 간단한 정리 순서를 지향합니다.
   - 수치/연도/전문 용어는 가능하면 구체적으로 제시합니다.
   - 사용자가 특별히 요청하지 않는 한, 지나치게 수학적/기술적 표기(복잡한 수식 등)는 피합니다.

4. reference 활용 방식
   - 여러 문서가 비슷한 내용을 말할 때는, 겹치는 핵심만 정리해 통합해서 설명하세요.
   - 서로 다른 관점을 제시하면, 그 사실을 드러내고 정리하세요.
     - "어떤 자료는 ~라고 설명하고, 다른 자료는 ~라고 설명합니다. 두 내용을 종합하면 …"
   - 문서 내용을 그대로 길게 복사/나열하지 말고, 요약·재구성해서 사용자 질문에 맞춰 답하세요.

5. 모르는 경우의 처리
   - 아래와 같은 경우에는 "모른다"고 말해야 합니다.
     - reference 어디에도 관련 정보가 없는 경우
     - 문서들이 서로 강하게 모순되어 있어 하나의 결론을 내기 어려운 경우
   - 이때는 다음처럼 답변합니다.
     - "제공된 자료에서는 이 질문에 대한 직접적인 정보를 찾을 수 없습니다."
     - "자료가 부족해 정확한 답을 드리기 어렵지만, 일반적으로 알려진 내용은 … 입니다."

[출력 형식]
- 사용자의 질문에 대한 자연스러운 한국어 답변 텍스트만 출력합니다.
- JSON 형식이나 메타데이터를 출력하지 않습니다.
- 답변 마지막에 한 문장으로 짧게 요약해 주면 좋습니다.

[검색 문서들]                                                                  
{reference}

[사용자 질문]
{input}

답변을 작성하세요.""", input_variables = ["input", "reference"])


        # rewrite_query 프롬프트
        logger.info(f"프롬프트 초기화 완료: {list(self._prompts.keys())}")




    def get_model(
        self,
        model_name: str = "solar-pro2",
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        지정된 모델명과 파라미터로 새로운 ChatUpstage 인스턴스를 생성하여 반환합니다.

        temperature와 추가 kwargs(예: stop sequences)를 지정할 수 있습니다.
        매번 새 인스턴스를 생성하므로 호출마다 다른 설정을 사용할 수 있습니다.

        Args:
            model_name: 모델 이름 (solar-pro2, solar-pro, solar-mini)
            temperature: 온도 설정 (None이면 기본값 사용)
            **kwargs: 추가 파라미터 (예: stop sequences)

        Returns:
            BaseChatModel: 새로 생성된 LLM 모델 인스턴스

        Raises:
            ValueError: 잘못된 모델 이름
        """
        if model_name not in self._models:
            raise ValueError(
                f"모델 '{model_name}'을 찾을 수 없습니다. "
                f"사용 가능한 모델: {list(self._models.keys())}"
            )

        # 새로운 파라미터로 모델 생성
        model_config = {
            "model": "solar-pro2" if model_name in ["solar-pro", "solar-pro2"] else "solar-mini",
            "upstage_api_key": Config.UPSTAGE_API_KEY
        }

        if temperature is not None:
            model_config["temperature"] = temperature
        else:
            model_config["temperature"] = 0 if model_name in ["solar-pro", "solar-pro2"] else 0.3

        # kwargs에서 추가 파라미터 병합 (예: stop)
        model_config.update(kwargs)

        return ChatUpstage(**model_config)



    def get_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        """
        프롬프트 템플릿을 반환합니다.

        Args:
            prompt_name: 프롬프트 이름

        Returns:
            ChatPromptTemplate: 프롬프트 템플릿

        Raises:
            ValueError: 이름이 잘못된 프롬프트 이름
        """
        if prompt_name not in self._prompts:
            raise ValueError(
                f"프롬프트 '{prompt_name}'을 찾을 수 없습니다. "
                f"사용 가능한 프롬프트: {list(self._prompts.keys())}"
            )

        return self._prompts[prompt_name]


# 싱글톤 인스턴스
_llm_manager_instance = None


def get_llm_manager() -> LLMManager:
    """LLM Manager 싱글톤 인스턴스를 반환합니다."""
    global _llm_manager_instance

    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()

    return _llm_manager_instance
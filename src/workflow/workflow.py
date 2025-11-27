import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Dict, List, Literal, Optional, TypedDict, Annotated
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


from src.agents.tools.tools import Retriver
from src.agents.query_cleaner import query_cleaner
from src.agents.science_cls import science_cls
from src.agents.generator import generator
from src.llm.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger


logger = get_logger(__name__)

class WorkflowState(TypedDict, total = False):
    question : str # 사용자의 질문
    standalone_query : str # 재작성된 쿼리
    messages : Annotated[list, add_messages] # chat history
    references : list
    topk : list
    answer : str # agent 답변
    cls : str


class Workflow:
    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.shared_llm = self.llm_manager.get_model(Config.LLM_MODEL, temperature = Config.LLM_TEMPERATURE)
        self.graph = self._build_graph()
        self.retriever = Retriver()


    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node('query_cleaner', self.query_clean_node)
        graph.add_node('retriever', self.retriever_node)
        graph.add_node('general_answer', self.general_answer_node)
        graph.add_node('answer',self.answer_node)

        graph.set_entry_point('query_cleaner')
        graph.add_conditional_edges(
            "query_cleaner",
            self._route_from_query_cleaner,
            {
                "not_science": 'general_answer',
                'science': 'retriever'
            }
        )
        graph.add_edge("retriever", 'answer')
        graph.add_edge("general_answer", END)
        graph.add_edge("answer", END)

        return graph.compile()
    
    def query_clean_node(self, state: WorkflowState) -> WorkflowState:
        """채팅기록과 읽어 사용자의 질문에 정확한 의도를 파악하여 query를 재작성하는 node"""
        query_rewrite_result = query_cleaner(state, llm = self.shared_llm)
        query = query_rewrite_result.get("rewritten_query")
        state['standalone_query'] = query
        return state
    
    def _route_from_query_cleaner(self, state: WorkflowState) -> WorkflowState:
        """standalone_query를 읽고 과학질문인지 아닌지 판단하는 node"""
        query = state['standalone_query']
        result = science_cls({'standalone_query' : query}, llm = self.shared_llm)
        return result
    def retriever_node(self, state: WorkflowState) -> WorkflowState:
        """standalone_query를 문서검색"""
        query = state.get('standalone_query')
        search_result = self.retriever.hybrid_retrieve(query_str=query, size = 3, alpha = 0)
        state['cls'] = 'science'
        retrieved_context = []
        references_temp = []
        topk_temp = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst['_source']['content'])
            references_temp.append({'score' : rst['_score'],
                                   'content' : rst['_source']['content']})
            topk_temp.append(rst['_source']['docid'])
        state['references'] = references_temp
        state['topk'] = topk_temp
        return state
    
    def answer_node(self, state: WorkflowState) -> WorkflowState:
        """reference를 참고하여 답안을 만드는 노드"""
        result = generator(state, llm = self.shared_llm)
        state['answer'] = result
        return state
    
    def general_answer_node(self, state:WorkflowState) -> WorkflowState:
        """일반답변"""
        question = state['standalone_query']
        result = self.shared_llm.invoke(question)
        state['answer'] = result.content
        state['cls'] = 'not_science'
        return state
    
    def run(self, question, messages = []):
        initial_state: WorkflowState = {
            'question' : question,
            'answer' : "",
            'messages' : messages,
            'references' : [],
            'standalone_query' : '',
            'topk' : [],
            'cls' : ''
        }

        result = self.graph.invoke(initial_state)

        return result
    
def build_workflow() -> Workflow:
    """외부에서 인스턴스 생성할 때 사용"""
    return Workflow()

if __name__ == "__main__" : 
    workflow = Workflow()
    result = workflow.run(question = '파이썬이 뭐야 ?')
    print(result)
    result1 = workflow.run(question = '너는 뭘 할 수 있어? ?')
    print(result1)
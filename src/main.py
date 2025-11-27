import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.eval import eval_rag
from src.workflow.workflow import build_workflow

if __name__ == "__main__":
    workflow = build_workflow()
    eval_rag(workflow=workflow, output_filename='test.csv')
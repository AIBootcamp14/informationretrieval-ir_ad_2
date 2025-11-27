import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from src.workflow.workflow import build_workflow
from src.utils.config import Config


def eval_rag(workflow, eval_filename = Config.EVAL_DATA_PATH, output_filename = 'submission.csv'):
    os.makedirs(os.path.join(Config.BASE_DIR, 'submission'), exist_ok=True)
    output_path= os.path.join(Config.BASE_DIR, 'submission',output_filename)

    with open(eval_filename, encoding = 'utf-8') as f, open(output_path, 'w', encoding = 'utf-8') as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j['msg']}')
            if len(j['msg']) == 1 :
                response = workflow.run(j['msg'])
            else :
                question = j['msg'][-1]
                messages = j['msg'][:-1]
                response = workflow.run(question = question, messages = messages)
            print(f'Answer: {response['answer']}\n')
            standalone_query = response['standalone_query'] if response['cls'] == 'science' else ''
            output = {
                "eval_id" : j['eval_id'],
                'standalone_query' : standalone_query,
                'topk' : response['topk'],
                'answer' : response['answer'],
                'references' : response['references']
            }
            of.write(json.dumps(output, ensure_ascii=False) + "\n")
            idx += 1

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import subprocess

from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Retriver:
    def __init__(self):
        self.ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
        self._ensure_elasticsearch_or_run_script(Config.SCRIPT_PATH)
        self.es = Elasticsearch(self.ES_HOST, request_timeout=30)
        self.index = Config.INDEX
        self.create_es_index(self.index, Config.SETTING, Config.MAPPINGS)
        self.embedding = SentenceTransformer(Config.EMBEDDING_MODEL)
        self._prepare_indexing(Config.DATA_PATH)

    def get_embedding(self, sentences):
        return self.embedding.encode(sentences)

    def get_embeddings_in_batches(self, docs, batch_size=100):
        batch_embeddings = []
        for i in tqdm(range(0, len(docs), batch_size), desc = "문서 Embeddings 변환 중"):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.get_embedding(contents)
            batch_embeddings.extend(embeddings)
        return batch_embeddings
    
    def create_es_index(self, index, settings, mappings):
        if self.es.indices.exists(index=index):
            self.es.indices.delete(index=index)
        self.es.indices.create(index=index, settings=settings, mappings=mappings)


    def delete_es_index(self,index):
        self.es.indices.delete(index=index)


    def bulk_add(self,index, docs):
        actions = [{"_index": index, "_source": doc} for doc in docs]
        return helpers.bulk(self.es, actions)


    def sparse_retrieve(self, query_str, size):
        query = {"match": {"content": {"query": query_str}}}
        return self.es.search(index="test", query=query, size=size, sort="_score")


    def dense_retrieve(self, query_str, size):
        query_embedding = self.get_embedding([query_str])[0]
        knn = {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": 100
        }
        return self.es.search(index="test", knn=knn)

    def hybrid_retrieve(self, query_str, size, alpha=0.5): # 하이브리드 함수
        """
        sparse(BM25)와 dense(KNN) 결과를 가중 합으로 섞는 하이브리드 검색.
        - alpha: sparse 가중치 (0~1). 0.5면 동등한 비중.
        """
        # 각각 검색
        sparse = self.sparse_retrieve(query_str, size)
        dense = self.dense_retrieve(query_str, size)
        
        combined = {}
        
        def normalized_and_add(results, weight):
            hits = results.get("hits", {}).get("hits", [])
            if not hits:
                return
            # 점수 정규화
            max_score = max(h["_score"] for h in hits) or 1.0
            for h in hits:
                src = h.get("_source", {})
                docid = src.get("docid")
                if docid is None:
                    # docid 없으면 스킵
                    continue
                norm_score = (h["_score"] / max_score) * weight
                if docid not in combined:
                    combined[docid] = {
                        "_source": src,
                        "_score": 0.0,
                    }
                combined[docid]["_score"] += norm_score
                
        # sparse / dense 각각 반영
        normalized_and_add(sparse, alpha)
        normalized_and_add(dense, 1 - alpha)
        
        # 점수 순으로 정렬 후 상위 size개 선택
        merged_hits = sorted(
            [
                {"_source": v["_source"], "_score": v["_score"]}
                for v in combined.values()
            ],
            key=lambda x: x["_score"],
            reverse=True,
        )[:size]
        
        # sparse_retrieve와 비슷한 형태로 반환
        return {"hits": {"hits": merged_hits}}
    

    def is_elasticsearch_running(self, host="http://localhost:9200"):
        try:
            es = Elasticsearch(hosts=[host])
            return es.ping()   # 정상 접속되면 True
        except Exception:
            return False
    
    def _ensure_elasticsearch_or_run_script(self,script_path, host="http://localhost:9200"):
        """ES가 실행 중이 아니면 bash 스크립트 실행"""
        
        if not self.is_elasticsearch_running(host):
            logger.info("[INFO] Elasticsearch가 실행 중이 아닙니다. 스크립트를 실행합니다...")
            
            try:
                subprocess.run(["bash", script_path], check=True)
                logger.info("es 실행 스크립트 실행 완료.")
            except subprocess.CalledProcessError as e:
                logger.error(f"[ERROR] 스크립트 실행 실패: {e}")
        else:
            logger.info("ElasticSearch가 이미 실행 중")

    def _prepare_indexing(self, data_path):
        index_docs = []
        with open(data_path, encoding = 'utf-8') as f:
            docs = [json.loads(line) for line in f]
        embeddings = self.get_embeddings_in_batches(docs)

        for doc, embedding in zip(docs, embeddings):
            doc['embeddings'] = embedding.tolist()
            index_docs.append(doc)
        ret = self.bulk_add(self.index, index_docs)
        logger.info("end prepare indexing")
        

if __name__ == "__main__":
    retriver = Retriver()
    a = retriver.hybrid_retrieve("파이썬 리스트에 대한 설명을 해줘", size = 3, alpha = .5)
    b = []
    for i in range(3):
        b.append({
        "content" : a['hits']['hits'][i]['_source']['content'],
        "score" : (a['hits']['hits'][i]['_score'])
        })
    print(b)
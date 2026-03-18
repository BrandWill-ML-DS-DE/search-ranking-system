from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.features.feature_builder import build_features
from src.ranking.lgbm_ranker import Ranker
from src.rerank.cross_encoder import Reranker

class SearchPipeline:
    def __init__(self, docs, model_path):
        self.bm25 = BM25Retriever(docs)
        self.dense = DenseRetriever(docs)
        self.ranker = Ranker(model_path)
        self.reranker = Reranker()

    def search(self, query):
        sparse = self.bm25.retrieve(query)
        dense = self.dense.retrieve(query)

        docs = list({d for d, _ in sparse + dense})

        feats = build_features(query, [(d, 1.0) for d in docs])
        scores = self.ranker.predict(feats)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:10]]

        final = self.reranker.rerank(query, top_docs)

        return final

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        emb = self.model.encode(docs)
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(np.array(emb))

    def retrieve(self, query, top_k=10):
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)
        return [(self.docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.doc_vecs = self.vectorizer.fit_transform(docs)

    def retrieve(self, query, top_k=10):
        q_vec = self.vectorizer.transform([query])
        scores = (self.doc_vecs @ q_vec.T).toarray().ravel()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], scores[i]) for i in idx]

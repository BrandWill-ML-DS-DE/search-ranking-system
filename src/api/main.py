from fastapi import FastAPI
from src.pipeline.search_pipeline import SearchPipeline

app = FastAPI()

docs = [
    "machine learning basics",
    "deep learning transformers",
    "cooking pasta",
    "neural networks explained"
]

pipeline = SearchPipeline(docs, "models/lgbm_ranker.txt")

@app.get("/search")
def search(q: str):
    results = pipeline.search(q)
    return {"results": results}

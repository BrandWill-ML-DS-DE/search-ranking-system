# 🔍 Enterprise Multi-Stage Search & Ranking System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ranking-FFD700)](https://lightgbm.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Dense_Retrieval-AA4A44)](https://github.com/facebookresearch/faiss)

A high-performance, end-to-end search architecture modeled after industry standards (Google, Amazon, LinkedIn). This system solves the **Retrieval-Augmented Generation (RAG)** and **Discovery** problem by implementing a multi-stage funnel that balances high recall with deep semantic precision.

---

## 🏗 System Architecture & Design Patterns

The project implements a classic **three-stage ranking funnel** to manage the trade-off between searching millions of documents and providing millisecond-level latency.

### 1. Hybrid Candidate Retrieval (Recall)

To maximize coverage, the system executes two parallel retrieval strategies:

* **Sparse Retrieval (BM25):** Handles exact keyword matching, acronyms, and lexical overlap using TF-IDF-based scoring.
* **Dense Retrieval (FAISS):** Captures deep semantic meaning using `all-MiniLM-L6-v2` embeddings.

### 2. Mid-Tier Ranking (Scoring)

Candidates from the retrieval stage are passed to a **LambdaMART (LightGBM)** ranker:

* Processes query-document features
* Optimized using **NDCG**

### 3. Final Re-ranking (Precision)

Top results are passed to a **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**:

* Joint query-document attention
* Highest precision stage

---

## 🛠 Tech Stack

| Component        | Technology            | Role                    |
| :--------------- | :-------------------- | :---------------------- |
| **Ranker**       | LightGBM              | LambdaRank optimization |
| **Vector Index** | FAISS                 | Dense similarity search |
| **Transformer**  | Sentence-Transformers | Retrieval + reranking   |
| **Serving**      | FastAPI               | API layer               |
| **DevOps**       | Docker                | Containerization        |

---

## 🚀 Engineering Highlights

* Modular `src/` architecture
* Multi-stage ranking pipeline
* Feature engineering layer
* YAML-based configuration
* Hybrid retrieval (sparse + dense)

---

## ⚙️ Local Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/search-ranking-system.git
cd search-ranking-system
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Train Ranking Model

```bash
python -m src.train
```

Generates:

```
models/lgbm_ranker.txt
```

---

### 5. Run API

```bash
uvicorn src.api.main:app --reload
```

---

### 6. Test Endpoint

Browser:

```
http://127.0.0.1:8000/search?q=machine+learning
```

Curl:

```bash
curl "http://127.0.0.1:8000/search?q=machine+learning"
```

---

### 7. API Docs

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Setup

### Build

```bash
docker build -t search-ranking-system .
```

### Run

```bash
docker run -p 8000:8000 search-ranking-system
```

---

## 📁 Project Structure

```
src/
├── api/
├── retrieval/
├── ranking/
├── rerank/
├── pipeline/
├── features/
├── train.py
```

---

## 🔄 End-to-End Flow

1. Query hits FastAPI
2. BM25 + Dense retrieval
3. Candidate merge
4. Feature generation
5. LambdaMART ranking
6. Cross-encoder reranking
7. Results returned

---

## ⚡ Performance Considerations

### Latency

* Fast retrieval first
* Expensive models last
* Top-K funnel design

### Memory

* In-memory embeddings
* Preloaded models

### Scalability

* FAISS → IVF / HNSW
* Feature store integration
* Horizontal API scaling

---

## 🧪 Evaluation Metrics

* **NDCG@K**
* **MAP**
* **Recall@K**
* Latency monitoring

---

## 🧩 Configuration

`config.yaml` enables tuning:

```yaml
retrieval:
  top_k: 20

rerank:
  top_k: 10
```

---

## 🚀 Future Roadmap

### Data & Scale

* [ ] MS MARCO full integration
* [ ] Streaming ingestion
* [ ] Distributed indexing

### Retrieval

* [ ] Hybrid score fusion
* [ ] ANN optimization (IVF/HNSW)
* [ ] Query expansion

### Ranking

* [ ] Advanced features
* [ ] Pairwise ranking
* [ ] Transformer rankers

### MLOps

* [ ] MLflow tracking
* [ ] Model registry
* [ ] CI/CD pipelines

### Online Learning

* [ ] Click modeling
* [ ] Personalization
* [ ] Reinforcement learning

### System Design

* [ ] Real-time indexing
* [ ] Redis caching
* [ ] Multi-tenant architecture

---

## 🎯 Why This Project Matters

* Demonstrates real-world ranking systems
* Shows retrieval + ranking separation
* Reflects production ML design patterns
* Goes beyond notebook-level ML

---

## 📌 Key Takeaway

This is a **production-style ML system**, not just a model.

Balances:

* Recall vs Precision
* Latency vs Accuracy
* Simplicity vs Scalability

---

## 👤 Author

Dr. Brandon Williams

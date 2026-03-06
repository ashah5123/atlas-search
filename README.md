# Atlas Search

Two-stage neural retrieval and LambdaRank reranking on **MS MARCO Passage Ranking**.  
Implements dense retrieval (FAISS + SentenceTransformer or trained bi-encoder) followed by feature-based Learning-to-Rank (LightGBM LambdaRank), with optional cross-encoder and stacked reranking.

This project demonstrates a production-style search stack with evaluation, latency benchmarking, and API serving.

---

## Architecture

```text
Query
  ↓
Dense Retrieval (MiniLM / Bi-Encoder → FAISS ANN)
  ↓
Candidate Set (Top-K)
  ↓
Feature Engineering
   - Dense similarity score
   - BM25 score
   - Query length
   - Passage length
   - Token overlap ratio
  ↓
LambdaRank (LightGBM)
  ↓
(Optional) Cross-Encoder Reranking
  ↓
FastAPI Search APIMulti-stage ranking separates:
	•	Recall optimization (dense retrieval)
	•	Precision optimization (reranking)
	•	Latency vs quality tradeoff

⸻

Results (MS MARCO 300k subset)
	•	Corpus size: 300k passages (subset of ~8.8M total)
	•	In-corpus dev queries: 16 (filtered from 2,000)

Retrieval (MiniLM + FAISS)
Metric
Value
Recall@10
0.8750
Recall@50
0.9375
Recall@100
Model
sentence-transformers/all-MiniLM-L6-v2
Reranking Ablation
System
Metrics
Value
LambdaRank (LightGBM)
NDCG@10 / MRR
0.6098 / 0.5080
Cross-encoder (ms-marco-MiniLM-L-6-v2)
NDCG@10 / MRR
0.5493 / 0.4439
Stacked (LambdaRank → Cross)
NDCG@10 / MRR
0.5955 / 0.5065

Tradeoffs
	•	LambdaRank: Best overall quality here with minimal latency overhead.
	•	Cross-encoder: Highest semantic modeling capacity but computationally expensive.
	•	Stacked: Production-style compromise (cheap first stage + expensive rerank on a short list).

Metrics are computed on an in-corpus dev subset because the system indexes a subset of MS MARCO.
Only dev queries whose relevant passages exist in the indexed subset are evaluated.

⸻

Performance (Latency Benchmark)

Local benchmark on Mac CPU, 20 requests, candidates_k=300, topk=10.
Reranker
p50 (ms)
p95 (ms)
none (dense only)
33.7
214.7
LambdaRank
35.1
43.5
Cross-encoder
1043.0
1245.5
Stacked
198.2
300.4

Observations
	•	LambdaRank adds minimal latency over dense retrieval.
	•	Cross-encoder is ~1s/query on CPU at 300 candidates.
	•	Stacked reranking demonstrates real-world latency/quality balancing.

⸻

Quickstart
# Python 3.11
pip install -r requirements.txt

# 1) Preprocess MS MARCO subset
python -m src.data.preprocess \
  --raw_dir data/raw/msmarco_passage \
  --out_dir data/processed/msmarco_passage \
  --max_passages 300000 \
  --max_train_queries 20000 \
  --max_dev_queries 2000

# 2) Build FAISS index
python -m src.indexing.build_faiss --config configs/config.yaml

# 3) Evaluate retrieval
python -m src.data.filter_dev_by_corpus --config configs/config.yaml
python -m src.eval.eval_retrieval --config configs/config.yaml

# 4) Build LTR dataset and train ranker
python -m src.ranking.build_ltr_dataset --config configs/config.yaml
python -m src.ranking.train_ranker --config configs/config.yaml
python -m src.eval.eval_rerank --config configs/config.yaml

# 5) Start API
uvicorn src.serving.app:app --reload

# In another terminal
./scripts/demo.sh
How to Run

1. Environment

Use Conda or venv with Python 3.11:conda create -n atlas-search python=3.11
conda activate atlas-search
pip install -r requirements.txt
2. Data

Download MS MARCO:
./scripts/download_msmarco.sh
Preprocess
python -m src.data.preprocess \
  --raw_dir data/raw/msmarco_passage \
  --out_dir data/processed/msmarco_passage \
  --max_passages 300000 \
  --max_train_queries 20000 \
  --max_dev_queries 2000
 3. Retrieval

(Optional) Train bi-encoder:
python -m src.indexing.build_faiss --config configs/config.yaml
Evaluate retrieval:
python -m src.eval.eval_retrieval --config configs/config.yaml
4. Reranking
python -m src.ranking.build_ltr_dataset --config configs/config.yaml
python -m src.ranking.train_ranker --config configs/config.yaml
python -m src.eval.eval_rerank --config configs/config.yaml
Optional cross-encoder + stacked evaluation:
python -m src.eval.eval_cross_rerank --config configs/config.yaml
python -m src.eval.eval_stacked_rerank --config configs/config.yaml
5. API and Demo
uvicorn src.serving.app:app --reload
Run demo:
./scripts/demo.sh
API Usage

Start server:
uvicorn src.serving.app:app --reload
Example query:
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what causes rainbows", "topk": 10, "candidates_k": 300, "reranker": "lgbm"}'
Available rerankers:
	•	"none"
	•	"lgbm" (default)
	•	"cross"
	•	"stacked"

Response format:
{
  "query": "...",
  "results": [
    {
      "pid": 123,
      "dense_score": 0.12,
      "rerank_score": 0.34,
      "cross_score": 1.23,
      "passage": "..."
    }
  ]
}
Technical Highlights
	•	Dense retrieval with transformer embeddings
	•	Approximate nearest neighbor indexing (FAISS)
	•	Feature-based Learning-to-Rank (LambdaRank / LightGBM)
	•	Cross-encoder reranking with joint attention
	•	Stacked reranking architecture
	•	Hard-negative mining experiments
	•	IR metric evaluation (Recall@K, NDCG@10, MRR)
	•	Latency benchmarking (p50 / p95)
	•	Production-ready FastAPI service
After overwriting, ensure the file is saved.

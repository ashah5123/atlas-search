# Atlas Search

Two-stage retrieval and LambdaRank reranking on **MS MARCO** passage ranking: dense retrieval (FAISS) with a SentenceTransformer or trained bi-encoder, then a LightGBM LambdaRank reranker over dense, BM25, and length/overlap features.

## Architecture

Query → **retrieval** (encode query, search FAISS) → **feature engineering** (dense score, BM25, length, overlap) → **LambdaRank** rerank → **FastAPI** search API.

## Results (300k MS MARCO Subset)

| Stage | Metric | Value |
|-------|--------|-------|
| MiniLM + FAISS | Recall@10 | 0.8750 |
| MiniLM + FAISS | Recall@50 | 0.9375 |
| MiniLM + FAISS | Recall@100 | 1.0000 |
| + LambdaRank | NDCG@10 | 0.6098 |
| + LambdaRank | MRR | 0.5080 |

Evaluation performed on in-corpus subset of MS MARCO dev queries (16 queries) when indexing 300,000 passages.

## Quickstart

```bash
# Python 3.11, pip install -r requirements.txt
# 1) Download data (see How to run), then preprocess
python -m src.data.preprocess --raw_dir data/raw/msmarco_passage --out_dir data/processed/msmarco_passage --max_passages 300000 --max_train_queries 20000 --max_dev_queries 2000

# 2) Build FAISS index
python -m src.indexing.build_faiss --config configs/config.yaml

# 3) Evaluate retrieval
python -m src.eval.eval_retrieval --config configs/config.yaml

# 4) LTR dataset, train ranker, eval rerank
python -m src.ranking.build_ltr_dataset --config configs/config.yaml
python -m src.ranking.train_ranker --config configs/config.yaml
python -m src.eval.eval_rerank --config configs/config.yaml

# 5) Start API and demo
uvicorn src.serving.app:app --reload
# In another terminal: ./scripts/demo.sh
```

## How to Run

1. **Environment**  
   Conda or venv with Python 3.11. `pip install -r requirements.txt`

2. **Data**  
   Download MS MARCO: `./scripts/download_msmarco.sh`  
   Preprocess:  
   `python -m src.data.preprocess --raw_dir data/raw/msmarco_passage --out_dir data/processed/msmarco_passage --max_passages 300000 --max_train_queries 20000 --max_dev_queries 2000`

3. **Retrieval**  
   (Optional) Train bi-encoder: `python -m src.models.train_retriever --config configs/config.yaml`  
   Build FAISS index: `python -m src.indexing.build_faiss --config configs/config.yaml`  
   Evaluate retrieval: `python -m src.eval.eval_retrieval --config configs/config.yaml`

4. **Reranking**  
   Build LTR dataset: `python -m src.ranking.build_ltr_dataset --config configs/config.yaml`  
   Train ranker: `python -m src.ranking.train_ranker --config configs/config.yaml`  
   Evaluate rerank: `python -m src.eval.eval_rerank --config configs/config.yaml`

5. **API and demo**  
   Start API: `uvicorn src.serving.app:app --reload`  
   Run demo: `./scripts/demo.sh` (expects API at http://127.0.0.1:8000)

## API Usage

Start the server with `uvicorn src.serving.app:app --reload`, then call the search endpoint:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "topk": 10, "candidates_k": 100}'
```

Response: `{"query": "...", "results": [{"pid": ..., "rerank_score": ..., "dense_score": ..., "passage": "..."}, ...]}`

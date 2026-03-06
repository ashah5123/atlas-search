# Atlas Search

Two-stage neural retrieval plus LambdaRank reranking on **MS MARCO** passage ranking: dense retrieval (FAISS) with an optional SentenceTransformer or trained bi-encoder, then a LightGBM LambdaRank reranker over dense + BM25 + length/overlap features.

## Architecture

Queries and passages are encoded with a bi-encoder (SentenceTransformer or trained HuggingFace model). Dense vectors are searched with **FAISS** (inner product). For each query, the top‑k candidates are turned into features (dense score, BM25, query/passage length, token overlap). A **LightGBM LambdaRank** model reranks by these features. The same pipeline is exposed via a **FastAPI** search API. Retrieval and rerank metrics are reported on an **in-corpus dev subset**: only dev queries and qrels whose passage IDs appear in the indexed corpus (e.g. when indexing a 500k subset of MS MARCO).

## Results (500k corpus)

| Setting | Value |
|--------|--------|
| Encoder | sentence-transformers/all-MiniLM-L6-v2 |
| Recall@10 | 0.7742 |
| Recall@50 | 0.9032 |
| Recall@100 | 0.9355 |
| NDCG@10 (reranked) | 0.6478 |
| MRR (reranked) | 0.5808 |

*All metrics are on the in-corpus dev subset (dev queries/qrels whose passages are in the 500k index).*

## Results (300k subset)

| Setting | Value |
|--------|--------|
| Corpus size | 300,000 passages (subset of MS MARCO Passage Ranking) |
| In-corpus dev | 16 queries (filtered from 2,000) |
| MiniLM+FAISS Recall@10 | 0.8750 |
| MiniLM+FAISS Recall@50 | 0.9375 |
| MiniLM+FAISS Recall@100 | 1.0000 |
| LambdaRank NDCG@10 | 0.6098 |
| LambdaRank MRR | 0.5080 |

Subset indexing requires in-corpus filtering of dev queries and qrels (only queries whose relevant passages appear in the index are evaluated). Larger `max_passages` increases dev coverage.

## Quickstart

```bash
# env: Python 3.11, pip install -r requirements.txt
# 1) Download data (see How to run), then preprocess
python -m src.data.preprocess --raw_dir data/raw/msmarco_passage --out_dir data/processed/msmarco_passage --max_passages 500000 --max_train_queries 20000 --max_dev_queries 2000

# 2) Build FAISS index
python -m src.indexing.build_faiss --config configs/config.yaml

# 3) Evaluate retrieval
python -m src.eval.eval_retrieval --config configs/config.yaml

# 4) LTR dataset + train ranker + eval rerank
python -m src.ranking.build_ltr_dataset --config configs/config.yaml
python -m src.ranking.train_ranker --config configs/config.yaml
python -m src.eval.eval_rerank --config configs/config.yaml

# 5) API + demo
uvicorn src.serving.app:app --reload
# in another terminal:
./scripts/demo.sh
```

## How to run

1. **Environment**
   - Conda (or venv) with Python 3.11.
   - `pip install -r requirements.txt`

2. **Data**
   - Download MS MARCO: `./scripts/download_msmarco.sh`
   - Preprocess:  
     `python -m src.data.preprocess --raw_dir data/raw/msmarco_passage --out_dir data/processed/msmarco_passage --max_passages 500000 --max_train_queries 20000 --max_dev_queries 2000`

3. **Retrieval**
   - (Optional) Train bi-encoder: `python -m src.models.train_retriever --config configs/config.yaml`
   - Build FAISS index: `python -m src.indexing.build_faiss --config configs/config.yaml`
   - Evaluate retrieval: `python -m src.eval.eval_retrieval --config configs/config.yaml`

4. **Reranking**
   - Build LTR dataset: `python -m src.ranking.build_ltr_dataset --config configs/config.yaml`
   - Train ranker: `python -m src.ranking.train_ranker --config configs/config.yaml`
   - Evaluate rerank: `python -m src.eval.eval_rerank --config configs/config.yaml`

5. **API and demo**
   - Start API: `uvicorn src.serving.app:app --reload`
   - Run demo: `./scripts/demo.sh` (expects API at http://127.0.0.1:8000)

# Atlas Search

Two-stage neural retrieval plus LambdaRank reranking on **MS MARCO** passage ranking: dense retrieval (FAISS) with an optional SentenceTransformer or trained bi-encoder, then a LightGBM LambdaRank reranker over dense + BM25 + length/overlap features.

## Results

| Setting | Value |
|--------|--------|
| Corpus size | 500,000 passages |
| Encoder backend | sentence-transformers/all-MiniLM-L6-v2 |
| Recall@10 | 0.7742 |
| Recall@50 | 0.9032 |
| Recall@100 | 0.9355 |
| NDCG@10 (reranked) | 0.6478 |
| MRR (reranked) | 0.5808 |

*Metrics are on an in-corpus dev subset when indexing a subset of MS MARCO.*

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

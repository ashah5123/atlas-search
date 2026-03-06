"""Evaluate stacked reranking: FAISS retrieval -> LambdaRank -> CrossEncoder rerank."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import faiss  # type: ignore[import]
import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from src.rerank.cross_encoder import CrossEncoderReranker
from src.utils.config import get_path, load_config

# Stability / reproducibility
faiss.omp_set_num_threads(1)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _load_dev_queries_in_corpus(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "dev_queries_in_corpus.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dev queries not found: {path}. Generate dev_queries_in_corpus.parquet first."
        )
    df = pd.read_parquet(path)
    print("Using dev queries: dev_queries_in_corpus.parquet")
    return df


def _load_dev_qrels_in_corpus(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "dev_qrels_in_corpus.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dev qrels not found: {path}. Generate dev_qrels_in_corpus.parquet first."
        )
    df = pd.read_parquet(path)
    print("Using dev qrels: dev_qrels_in_corpus.parquet")
    return df


def _build_relevant_map(qrels: pd.DataFrame) -> Dict[int, set[int]]:
    """qid -> set of pids with relevance > 0."""
    pos = qrels[qrels["relevance"] > 0]
    out: Dict[int, set[int]] = {}
    for _, row in pos.iterrows():
        qid = int(row["qid"])
        pid = int(row["pid"])
        out.setdefault(qid, set()).add(pid)
    return out


def _ndcg_at_k(relevances: List[int], k: int = 10) -> float:
    """Binary relevance NDCG@k. relevances is the list of 0/1 in ranked order (top k)."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevances))
    num_pos = sum(relevances)
    if num_pos == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, num_pos)))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(reranked_pids: List[int], rel_pids: set[int], k: int) -> float:
    """Reciprocal rank of first relevant doc in top k; 0 if none."""
    for i, pid in enumerate(reranked_pids[:k]):
        if pid in rel_pids:
            return 1.0 / (i + 1)
    return 0.0


def _load_faiss_index(repo_root: Path) -> tuple[faiss.Index, np.ndarray]:
    index_path = repo_root / "artifacts" / "index" / "faiss.index"
    pids_path = repo_root / "artifacts" / "index" / "pids.npy"
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run src.indexing.build_faiss first."
        )
    if not pids_path.exists():
        raise FileNotFoundError(
            f"PIDs mapping not found: {pids_path}. Run src.indexing.build_faiss first."
        )
    index = faiss.read_index(str(index_path))
    pids = np.load(pids_path)
    return index, pids


def _load_passages_for_index(processed_dir: Path, pids: np.ndarray) -> Dict[int, str]:
    path = processed_dir / "passages.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Passages not found: {path}. Run preprocessing first.")
    df = pd.read_parquet(path)
    if "pid" not in df.columns or "passage" not in df.columns:
        raise ValueError(
            f"passages.parquet must have 'pid', 'passage'; found {df.columns.tolist()}"
        )
    pid_set = set(int(x) for x in pids.tolist())
    df = df[df["pid"].isin(pid_set)]
    mapping = (
        df[["pid", "passage"]]
        .drop_duplicates("pid")
        .set_index("pid")["passage"]
        .astype(str)
        .to_dict()
    )
    return {int(k): str(v) for k, v in mapping.items()}


def run_eval_stacked_rerank(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))
    repo_root = Path(__file__).resolve().parents[2]

    faiss_cfg: Dict[str, Any] = cfg.get("faiss", {})
    serving_cfg: Dict[str, Any] = cfg.get("serving", {})
    stacked_cfg: Dict[str, Any] = cfg.get("stacked_rerank", {})
    cross_cfg: Dict[str, Any] = cfg.get("cross_encoder", {})

    st_model_name = faiss_cfg.get("st_model_name")
    if not st_model_name:
        raise KeyError("Config must define 'faiss.st_model_name' for MiniLM+FAISS retrieval.")

    candidates_k = int(serving_cfg.get("candidates_k", 100))
    stack_topn = int(stacked_cfg.get("topn", 50))
    cross_model_name = str(cross_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    cross_batch_size = int(cross_cfg.get("batch_size", 32))

    dev_queries = _load_dev_queries_in_corpus(processed_dir)
    dev_qrels = _load_dev_qrels_in_corpus(processed_dir)
    for col in ("qid", "query"):
        if col not in dev_queries.columns:
            raise ValueError(f"Dev queries must have 'qid', 'query'; found {dev_queries.columns.tolist()}")
    for col in ("qid", "pid", "relevance"):
        if col not in dev_qrels.columns:
            raise ValueError(f"Dev qrels must have 'qid', 'pid', 'relevance'; found {dev_qrels.columns.tolist()}")

    relevant_by_qid = _build_relevant_map(dev_qrels)

    # Ranker assets
    ranker_path = repo_root / "artifacts" / "ranker" / "lgbm_ranker.txt"
    names_path = repo_root / "artifacts" / "ranker" / "feature_names.json"
    if not ranker_path.exists():
        raise FileNotFoundError(
            f"Ranker not found: {ranker_path}. Run src.ranking.train_ranker first."
        )
    if not names_path.exists():
        raise FileNotFoundError(
            f"Feature names not found: {names_path}. Run src.ranking.train_ranker first."
        )
    with open(names_path) as f:
        feature_names = json.load(f)
    ranker = lgb.Booster(model_file=str(ranker_path))

    index, pids = _load_faiss_index(repo_root)
    passage_by_pid = _load_passages_for_index(processed_dir, pids)
    k = min(candidates_k, int(index.ntotal))
    if k <= 0:
        raise RuntimeError("FAISS index is empty.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "eval_stacked_rerank requires 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        ) from exc

    st_model = SentenceTransformer(st_model_name)
    cross = CrossEncoderReranker(model_name=cross_model_name, device=None, batch_size=cross_batch_size)

    qids_out: List[int] = []
    retrieved_pids_out: List[List[int]] = []
    dense_scores_out: List[List[float]] = []
    rerank_scores_out: List[List[float]] = []
    cross_scores_out: List[List[float]] = []
    ndcg_scores: List[float] = []
    mrr_scores: List[float] = []

    for _, row in tqdm(dev_queries.iterrows(), total=len(dev_queries), desc="Stacked rerank"):
        qid = int(row["qid"])
        query_text = str(row["query"])
        rel_pids = relevant_by_qid.get(qid, set())

        q_emb = st_model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.shape[1] != index.d:
            raise ValueError(
                f"Query embedding dim ({q_emb.shape[1]}) != FAISS index dim ({index.d}). "
                "Use the same encoder as the index (faiss.st_model_name)."
            )

        dense_scores, indices = index.search(q_emb, k)
        dense_scores = dense_scores[0].tolist()
        indices = indices[0]
        cand_pids = pids[indices].tolist()

        # Build candidate passages (skip missing)
        cand_passages: List[str] = []
        valid_pids: List[int] = []
        valid_dense: List[float] = []
        for pid, ds in zip(cand_pids, dense_scores):
            pid_int = int(pid)
            text = passage_by_pid.get(pid_int)
            if text is None:
                continue
            valid_pids.append(pid_int)
            valid_dense.append(float(ds))
            cand_passages.append(text)

        if not valid_pids:
            continue

        # LambdaRank features (same as serving app)
        query_tokens = _tokenize(query_text)
        q_len = len(query_tokens)
        docs_tokens = [_tokenize(p) for p in cand_passages]
        bm25 = BM25Okapi(docs_tokens)
        bm25_scores = bm25.get_scores(query_tokens).tolist()

        rows: List[Dict[str, float]] = []
        for i in range(len(valid_pids)):
            p_tokens = docs_tokens[i]
            overlap = len(set(query_tokens) & set(p_tokens))
            overlap_ratio = overlap / max(1, len(query_tokens))
            rows.append({
                "dense_score": valid_dense[i],
                "bm25_score": float(bm25_scores[i]),
                "query_len": q_len,
                "passage_len": len(p_tokens),
                "token_overlap_ratio": overlap_ratio,
            })
        X = pd.DataFrame(rows)[feature_names]
        rerank_pred = ranker.predict(X)

        # Sort by LambdaRank and keep top stack_topn
        order = np.argsort(-np.asarray(rerank_pred))
        topn = min(max(1, stack_topn), len(order))
        lambda_items: List[dict] = []
        for idx in order[:topn]:
            lambda_items.append({
                "pid": valid_pids[idx],
                "dense_score": valid_dense[idx],
                "rerank_score": float(rerank_pred[idx]),
                "passage": cand_passages[idx],
            })

        # Cross-encoder rerank of top-N
        cross_items = cross.rerank(query_text, lambda_items, text_key="passage")

        final_pids = [int(x["pid"]) for x in cross_items]
        final_dense = [float(x["dense_score"]) for x in cross_items]
        final_rerank = [float(x["rerank_score"]) for x in cross_items]
        final_cross = [float(x["cross_score"]) for x in cross_items]

        qids_out.append(qid)
        retrieved_pids_out.append(final_pids)
        dense_scores_out.append(final_dense)
        rerank_scores_out.append(final_rerank)
        cross_scores_out.append(final_cross)

        rel_vector = [1 if p in rel_pids else 0 for p in final_pids]
        ndcg_scores.append(_ndcg_at_k(rel_vector, k=10))
        mrr_scores.append(_mrr_at_k(final_pids, rel_pids, k=len(final_pids)))

    if not qids_out:
        raise RuntimeError("No queries were reranked. Check dev data and index.")

    ndcg10 = float(np.mean(ndcg_scores))
    mrr = float(np.mean(mrr_scores))

    print(f"NDCG@10 (stacked): {ndcg10:.4f}")
    print(f"MRR (stacked): {mrr:.4f}")

    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "stacked_reranked_results.parquet"

    out_df = pd.DataFrame({
        "qid": qids_out,
        "retrieved_pids": retrieved_pids_out,
        "dense_scores": dense_scores_out,
        "rerank_scores": rerank_scores_out,
        "cross_scores": cross_scores_out,
    })
    out_df["qid"] = out_df["qid"].astype("int64")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved stacked reranked results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stacked reranking: LambdaRank + CrossEncoder.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_eval_stacked_rerank(args.config)


if __name__ == "__main__":
    main()


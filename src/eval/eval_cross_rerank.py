"""Evaluate cross-encoder reranking: FAISS retrieval + cross-encoder rerank, NDCG@10 and MRR."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import faiss  # type: ignore[import]
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.rerank.cross_encoder import CrossEncoderReranker
from src.utils.config import get_path, load_config

faiss.omp_set_num_threads(1)


def _load_dev_queries(processed_dir: Path) -> pd.DataFrame:
    """Load dev queries, preferring dev_queries_in_corpus.parquet."""
    for name in ("dev_queries_in_corpus.parquet", "dev_queries.parquet"):
        p = processed_dir / name
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Using dev queries: {name}")
            return df
    raise FileNotFoundError(
        f"No dev queries found in {processed_dir}. "
        "Run preprocessing (and optionally filter_dev_by_corpus) first."
    )


def _load_dev_qrels(processed_dir: Path) -> pd.DataFrame:
    """Load dev qrels, preferring dev_qrels_in_corpus.parquet."""
    for name in ("dev_qrels_in_corpus.parquet", "dev_qrels.parquet"):
        p = processed_dir / name
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Using dev qrels: {name}")
            return df
    raise FileNotFoundError(
        f"No dev qrels found in {processed_dir}. "
        "Run preprocessing (and optionally filter_dev_by_corpus) first."
    )


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


def _mrr_at_k(reranked_pids: List[int], rel_pids: set[int], k: int = 10) -> float:
    """Reciprocal rank of first relevant doc in top k; 0 if none."""
    for i, pid in enumerate(reranked_pids[:k]):
        if pid in rel_pids:
            return 1.0 / (i + 1)
    return 0.0


def _load_faiss_index(repo_root: Path) -> tuple[faiss.Index, np.ndarray]:
    """Load FAISS index and pids from artifacts/index/."""
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
    """Load passages.parquet and return pid -> passage for pids in index."""
    path = processed_dir / "passages.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Passages not found: {path}. Run preprocessing first.")
    df = pd.read_parquet(path)
    if "pid" not in df.columns or "passage" not in df.columns:
        raise ValueError(f"passages.parquet must have 'pid', 'passage'; found {df.columns.tolist()}")
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


def run_eval_cross_rerank(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))
    repo_root = Path(__file__).resolve().parents[2]

    retriever_cfg = cfg.get("retriever", {})
    faiss_cfg = cfg.get("faiss", {})
    cross_cfg = cfg.get("cross_encoder", {})

    candidates_k = int(
        retriever_cfg.get("candidates_k") or faiss_cfg.get("topk", 100)
    )
    cross_model_name = str(
        cross_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    cross_batch_size = int(cross_cfg.get("batch_size", 32))

    st_model_name = faiss_cfg.get("st_model_name")
    if not st_model_name:
        raise KeyError(
            "faiss.st_model_name is required for MiniLM retrieval. "
            "Set it in config (e.g. sentence-transformers/all-MiniLM-L6-v2)."
        )

    dev_queries = _load_dev_queries(processed_dir)
    dev_qrels = _load_dev_qrels(processed_dir)
    for col in ("qid", "query"):
        if col not in dev_queries.columns:
            raise ValueError(
                f"Dev queries must have 'qid', 'query'; found {dev_queries.columns.tolist()}"
            )
    for col in ("qid", "pid", "relevance"):
        if col not in dev_qrels.columns:
            raise ValueError(
                f"Dev qrels must have 'qid', 'pid', 'relevance'; found {dev_qrels.columns.tolist()}"
            )

    index, pids = _load_faiss_index(repo_root)
    passage_by_pid = _load_passages_for_index(processed_dir, pids)
    relevant_by_qid = _build_relevant_map(dev_qrels)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "eval_cross_rerank requires 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        ) from exc

    st_model = SentenceTransformer(st_model_name)
    cross_reranker = CrossEncoderReranker(
        model_name=cross_model_name,
        device=None,
        batch_size=cross_batch_size,
    )

    k = min(candidates_k, index.ntotal)
    if k <= 0:
        raise RuntimeError("FAISS index is empty.")

    qids_out: List[int] = []
    retrieved_pids_out: List[List[int]] = []
    dense_scores_out: List[List[float]] = []
    cross_scores_out: List[List[float]] = []
    ndcg_scores: List[float] = []
    mrr_scores: List[float] = []

    for _, row in tqdm(
        dev_queries.iterrows(), total=len(dev_queries), desc="Cross-encoder rerank"
    ):
        qid = int(row["qid"])
        query_text = str(row["query"])

        q_emb = st_model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.shape[1] != index.d:
            raise ValueError(
                f"Query embedding dim ({q_emb.shape[1]}) != FAISS index dim ({index.d}). "
                "Use the same encoder as the index (faiss.st_model_name)."
            )

        scores, indices = index.search(q_emb, k)
        scores = scores[0]
        indices = indices[0]
        cand_pids = pids[indices].tolist()
        cand_dense = scores.tolist()

        candidates: List[Dict[str, Any]] = []
        for pid, d in zip(cand_pids, cand_dense):
            pid_int = int(pid)
            text = passage_by_pid.get(pid_int)
            if text is None:
                continue
            candidates.append({
                "pid": pid_int,
                "dense_score": float(d),
                "passage": text,
            })

        if not candidates:
            continue

        reranked = cross_reranker.rerank(query_text, candidates, text_key="passage")

        retrieved_pids = [c["pid"] for c in reranked]
        dense_scores = [c["dense_score"] for c in reranked]
        cross_scores = [c["cross_score"] for c in reranked]

        qids_out.append(qid)
        retrieved_pids_out.append(retrieved_pids)
        dense_scores_out.append(dense_scores)
        cross_scores_out.append(cross_scores)

        rel_pids = relevant_by_qid.get(qid, set())
        rel_vector = [1 if p in rel_pids else 0 for p in retrieved_pids]
        ndcg_scores.append(_ndcg_at_k(rel_vector, k=10))
        mrr_scores.append(_mrr_at_k(retrieved_pids, rel_pids, k=10))

    if not qids_out:
        raise RuntimeError("No queries were reranked. Check dev data and index.")

    ndcg10 = float(np.mean(ndcg_scores))
    mrr = float(np.mean(mrr_scores))

    print(f"NDCG@10 (cross-encoder): {ndcg10:.4f}")
    print(f"MRR (cross-encoder): {mrr:.4f}")

    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "cross_reranked_results.parquet"

    out_df = pd.DataFrame({
        "qid": qids_out,
        "retrieved_pids": retrieved_pids_out,
        "dense_scores": dense_scores_out,
        "cross_scores": cross_scores_out,
    })
    out_df["qid"] = out_df["qid"].astype("int64")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved cross-encoder reranked results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-encoder reranking after FAISS retrieval."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_eval_cross_rerank(args.config)


if __name__ == "__main__":
    main()

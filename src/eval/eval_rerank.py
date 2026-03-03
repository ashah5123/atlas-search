"""Evaluate reranking with the trained LightGBM ranker and compute NDCG@10 and MRR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from src.utils.config import get_path, load_config


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


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
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevances))
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


def run_eval_rerank(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))
    repo_root = Path(__file__).resolve().parents[2]

    # Inputs
    retrieval_path = repo_root / "artifacts" / "eval" / "retrieval_results.parquet"
    if not retrieval_path.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {retrieval_path}. Run src.eval.eval_retrieval first."
        )
    retrieval_df = pd.read_parquet(retrieval_path)
    for col in ("qid", "retrieved_pids", "scores"):
        if col not in retrieval_df.columns:
            raise ValueError(
                f"retrieval_results.parquet must have columns qid, retrieved_pids, scores; "
                f"found {retrieval_df.columns.tolist()}"
            )

    dev_queries = _load_dev_queries(processed_dir)
    dev_qrels = _load_dev_qrels(processed_dir)
    for col in ("qid", "query"):
        if col not in dev_queries.columns:
            raise ValueError(f"Dev queries must have 'qid', 'query'; found {dev_queries.columns.tolist()}")
    for col in ("qid", "pid", "relevance"):
        if col not in dev_qrels.columns:
            raise ValueError(f"Dev qrels must have 'qid', 'pid', 'relevance'; found {dev_qrels.columns.tolist()}")

    passages_path = processed_dir / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages not found: {passages_path}. Run preprocessing first.")
    passages = pd.read_parquet(passages_path)
    if "pid" not in passages.columns or "passage" not in passages.columns:
        raise ValueError(f"passages.parquet must have 'pid', 'passage'; found {passages.columns.tolist()}")

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
    model = lgb.Booster(model_file=str(ranker_path))

    query_by_qid: Dict[int, str] = (
        dev_queries[["qid", "query"]]
        .drop_duplicates("qid")
        .set_index("qid")["query"]
        .astype(str)
        .to_dict()
    )
    passage_by_pid: Dict[int, str] = (
        passages[["pid", "passage"]]
        .drop_duplicates("pid")
        .set_index("pid")["passage"]
        .astype(str)
        .to_dict()
    )
    relevant_by_qid = _build_relevant_map(dev_qrels)

    qids_out: List[int] = []
    reranked_pids_out: List[List[int]] = []
    reranked_scores_out: List[List[float]] = []
    ndcg_scores: List[float] = []
    mrr_scores: List[float] = []

    for _, row in tqdm(retrieval_df.iterrows(), total=len(retrieval_df), desc="Reranking"):
        qid = int(row["qid"])
        cand_pids: List[int] = [int(p) for p in row["retrieved_pids"]]
        cand_scores: List[float] = [float(s) for s in row["scores"]]

        if not cand_pids or qid not in query_by_qid:
            continue

        query_text = query_by_qid[qid]
        query_tokens = _tokenize(query_text)
        q_len = len(query_tokens)
        rel_pids = relevant_by_qid.get(qid, set())

        docs_tokens: List[List[str]] = []
        valid_pids: List[int] = []
        valid_dense: List[float] = []
        for pid, score in zip(cand_pids, cand_scores):
            text = passage_by_pid.get(pid)
            if text is None:
                continue
            docs_tokens.append(_tokenize(text))
            valid_pids.append(pid)
            valid_dense.append(score)

        if not docs_tokens:
            continue

        bm25 = BM25Okapi(docs_tokens)
        bm25_list = bm25.get_scores(query_tokens).tolist()

        rows: List[Dict[str, float]] = []
        for i in range(len(valid_pids)):
            p_tokens = docs_tokens[i]
            overlap = len(set(query_tokens) & set(p_tokens))
            overlap_ratio = overlap / max(1, len(query_tokens))
            rows.append({
                "dense_score": valid_dense[i],
                "bm25_score": float(bm25_list[i]),
                "query_len": q_len,
                "passage_len": len(p_tokens),
                "token_overlap_ratio": overlap_ratio,
            })
        X = pd.DataFrame(rows)[feature_names]
        pred = model.predict(X)

        order = np.argsort(-np.asarray(pred))
        reranked_pids = [valid_pids[i] for i in order]
        reranked_scores = [float(pred[i]) for i in order]

        qids_out.append(qid)
        reranked_pids_out.append(reranked_pids)
        reranked_scores_out.append(reranked_scores)

        rel_vector = [1 if p in rel_pids else 0 for p in reranked_pids]
        ndcg_scores.append(_ndcg_at_k(rel_vector, k=10))
        mrr_scores.append(_mrr_at_k(reranked_pids, rel_pids, k=len(reranked_pids)))

    if not qids_out:
        raise RuntimeError("No queries were reranked. Check retrieval_results and dev data.")

    ndcg10 = float(np.mean(ndcg_scores))
    mrr = float(np.mean(mrr_scores))

    print(f"NDCG@10 (reranked): {ndcg10:.4f}")
    print(f"MRR (reranked): {mrr:.4f}")

    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "reranked_results.parquet"

    out_df = pd.DataFrame({
        "qid": qids_out,
        "reranked_pids": reranked_pids_out,
        "reranked_scores": reranked_scores_out,
    })
    out_df["qid"] = out_df["qid"].astype("int64")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved reranked results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reranking with LightGBM ranker.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_eval_rerank(args.config)


if __name__ == "__main__":
    main()

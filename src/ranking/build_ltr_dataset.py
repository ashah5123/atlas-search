"""Build a learning-to-rank dataset from retrieval results and dev qrels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from src.utils.config import get_path, load_config


def _load_dev_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dev queries and qrels, preferring *_in_corpus variants if present."""
    queries_corpus = processed_dir / "dev_queries_in_corpus.parquet"
    queries_default = processed_dir / "dev_queries.parquet"
    qrels_corpus = processed_dir / "dev_qrels_in_corpus.parquet"
    qrels_default = processed_dir / "dev_qrels.parquet"

    queries_path = queries_corpus if queries_corpus.exists() else queries_default
    qrels_path = qrels_corpus if qrels_corpus.exists() else qrels_default

    if not queries_path.exists():
        raise FileNotFoundError(
            f"Dev queries not found: {queries_path}. "
            "Run preprocessing (and optionally filter_dev_by_corpus) first."
        )
    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Dev qrels not found: {qrels_path}. "
            "Run preprocessing (and optionally filter_dev_by_corpus) first."
        )

    print(f"Using dev queries: {queries_path.name}")
    print(f"Using dev qrels: {qrels_path.name}")

    queries = pd.read_parquet(queries_path)
    qrels = pd.read_parquet(qrels_path)

    for col in ("qid", "query"):
        if col not in queries.columns:
            raise ValueError(
                f"Dev queries must have columns 'qid', 'query'; found {queries.columns.tolist()}"
            )
    for col in ("qid", "pid", "relevance"):
        if col not in qrels.columns:
            raise ValueError(
                f"Dev qrels must have columns 'qid', 'pid', 'relevance'; found {qrels.columns.tolist()}"
            )

    return queries, qrels


def _load_retrieval_results(repo_root: Path) -> pd.DataFrame:
    """Load retrieval results parquet with qid, retrieved_pids, scores."""
    path = repo_root / "artifacts" / "eval" / "retrieval_results.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {path}. "
            "Run src.eval.eval_retrieval first."
        )
    df = pd.read_parquet(path)
    for col in ("qid", "retrieved_pids", "scores"):
        if col not in df.columns:
            raise ValueError(
                f"retrieval_results.parquet must have columns 'qid', 'retrieved_pids', 'scores'; "
                f"found {df.columns.tolist()}"
            )
    return df


def _build_relevant_map(qrels: pd.DataFrame) -> Dict[int, set[int]]:
    """Build qid -> set of relevant pids (relevance > 0)."""
    pos = qrels[qrels["relevance"] > 0]
    rel: Dict[int, set[int]] = {}
    for _, row in pos.iterrows():
        qid = int(row["qid"])
        pid = int(row["pid"])
        rel.setdefault(qid, set()).add(pid)
    return rel


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_ltr_dataset(config_path: str) -> None:
    """Construct an LTR dataset from retrieval results and dev qrels."""
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))

    repo_root = Path(__file__).resolve().parents[2]
    ltr_dir = repo_root / "data" / "processed" / "ltr"
    ltr_dir.mkdir(parents=True, exist_ok=True)

    retrieval_df = _load_retrieval_results(repo_root)
    dev_queries, dev_qrels = _load_dev_data(processed_dir)

    passages_path = processed_dir / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages parquet not found: {passages_path}. Run preprocessing first."
        )
    passages = pd.read_parquet(passages_path)
    if "pid" not in passages.columns or "passage" not in passages.columns:
        raise ValueError(
            f"passages.parquet must have columns 'pid', 'passage'; found {passages.columns.tolist()}"
        )

    # Index lookups
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

    # Containers for LTR rows and group sizes
    qid_list: List[int] = []
    pid_list: List[int] = []
    labels: List[int] = []
    dense_scores: List[float] = []
    bm25_scores: List[float] = []
    query_lens: List[int] = []
    passage_lens: List[int] = []
    token_overlap_ratios: List[float] = []
    groups: List[int] = []

    for _, row in tqdm(retrieval_df.iterrows(), total=len(retrieval_df), desc="Building LTR rows"):
        qid = int(row["qid"])
        cand_pids: List[int] = list(row["retrieved_pids"])
        cand_scores: List[float] = list(row["scores"])

        if not cand_pids:
            continue

        if qid not in query_by_qid:
            # Skip queries not present in dev_queries
            continue
        query_text = query_by_qid[qid]
        query_tokens = _tokenize(query_text)
        q_len = len(query_tokens)
        rel_pids = relevant_by_qid.get(qid, set())

        # Collect passage texts for this query's candidates
        docs_tokens: List[List[str]] = []
        docs_texts: List[str] = []
        valid_indices: List[int] = []
        for idx, pid in enumerate(cand_pids):
            pid_int = int(pid)
            passage_text = passage_by_pid.get(pid_int)
            if passage_text is None:
                continue
            tokens = _tokenize(passage_text)
            docs_tokens.append(tokens)
            docs_texts.append(passage_text)
            valid_indices.append(idx)

        if not docs_tokens:
            continue

        # BM25 over candidate passages for this query
        bm25 = BM25Okapi(docs_tokens)
        bm25_scores_query = bm25.get_scores(query_tokens).tolist()

        group_size = len(valid_indices)
        groups.append(group_size)

        for local_idx, orig_idx in enumerate(valid_indices):
            pid_int = int(cand_pids[orig_idx])
            passage_text = docs_texts[local_idx]
            p_tokens = docs_tokens[local_idx]

            label = 1 if pid_int in rel_pids else 0
            dense_score = float(cand_scores[orig_idx])
            bm25_score = float(bm25_scores_query[local_idx])
            p_len = len(p_tokens)
            overlap = len(set(query_tokens) & set(p_tokens))
            overlap_ratio = overlap / max(1, len(query_tokens))

            qid_list.append(qid)
            pid_list.append(pid_int)
            labels.append(label)
            dense_scores.append(dense_score)
            bm25_scores.append(bm25_score)
            query_lens.append(q_len)
            passage_lens.append(p_len)
            token_overlap_ratios.append(overlap_ratio)

    if not qid_list:
        raise RuntimeError("No LTR rows were generated. Check retrieval_results and dev data alignment.")

    dev_ltr = pd.DataFrame(
        {
            "qid": qid_list,
            "pid": pid_list,
            "label": labels,
            "dense_score": dense_scores,
            "bm25_score": bm25_scores,
            "query_len": query_lens,
            "passage_len": passage_lens,
            "token_overlap_ratio": token_overlap_ratios,
        }
    )
    dev_ltr["qid"] = dev_ltr["qid"].astype("int64")
    dev_ltr["pid"] = dev_ltr["pid"].astype("int64")
    dev_ltr["label"] = dev_ltr["label"].astype("int64")

    dev_groups = np.asarray(groups, dtype=np.int64)

    out_parquet = ltr_dir / "dev_ltr.parquet"
    out_groups = ltr_dir / "dev_groups.npy"
    dev_ltr.to_parquet(out_parquet, index=False)
    np.save(out_groups, dev_groups)

    num_queries = len(groups)
    total_candidates = len(dev_ltr)
    positive_labels = int(dev_ltr["label"].sum())

    print(f"Number of queries: {num_queries:,}")
    print(f"Total candidates: {total_candidates:,}")
    print(f"Positive labels: {positive_labels:,}")
    print(f"dev_ltr shape: {dev_ltr.shape}")
    print(f"Saved LTR dataset to {out_parquet}")
    print(f"Saved group sizes to {out_groups}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learning-to-rank dataset from retrieval results.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    build_ltr_dataset(args.config)


if __name__ == "__main__":
    main()


"""Evaluate retrieval: encode dev queries, run FAISS search, compute Recall@K."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import faiss  # type: ignore[import]
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.config import get_path, load_config

# Limit FAISS to a single thread for reproducibility / stability.
faiss.omp_set_num_threads(1)
print("FAISS threads: 1")


def _load_dev_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dev queries and qrels Parquet files. Prefer *_in_corpus versions if present."""
    queries_corpus = processed_dir / "dev_queries_in_corpus.parquet"
    queries_default = processed_dir / "dev_queries.parquet"
    qrels_corpus = processed_dir / "dev_qrels_in_corpus.parquet"
    qrels_default = processed_dir / "dev_qrels.parquet"

    if queries_corpus.exists():
        queries_path = queries_corpus
    else:
        queries_path = queries_default
    if qrels_corpus.exists():
        qrels_path = qrels_corpus
    else:
        qrels_path = qrels_default

    if not queries_path.exists():
        raise FileNotFoundError(
            f"Dev queries not found: {queries_path}. "
            "Run preprocessing to generate dev_queries.parquet (or filter_dev_by_corpus for dev_queries_in_corpus.parquet)."
        )
    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Dev qrels not found: {qrels_path}. "
            "Run preprocessing to generate dev_qrels.parquet (or filter_dev_by_corpus for dev_qrels_in_corpus.parquet)."
        )

    print(f"Using dev queries: {queries_path.name}")
    print(f"Using dev qrels: {qrels_path.name}")

    queries = pd.read_parquet(queries_path)
    qrels = pd.read_parquet(qrels_path)

    for col in ("qid", "query"):
        if col not in queries.columns:
            raise ValueError(f"dev_queries must have columns 'qid', 'query'; found {queries.columns.tolist()}")
    for col in ("qid", "pid", "relevance"):
        if col not in qrels.columns:
            raise ValueError(f"dev_qrels must have columns 'qid', 'pid', 'relevance'; found {qrels.columns.tolist()}")

    return queries, qrels


def _load_faiss_index(repo_root: Path) -> tuple[faiss.Index, np.ndarray]:
    """Load FAISS index and pids mapping from artifacts/index/."""
    index_path = repo_root / "artifacts" / "index" / "faiss.index"
    pids_path = repo_root / "artifacts" / "index" / "pids.npy"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. "
            "Run src.indexing.build_faiss to build the index first."
        )
    if not pids_path.exists():
        raise FileNotFoundError(
            f"PIDs mapping not found: {pids_path}. "
            "Run src.indexing.build_faiss to generate pids.npy."
        )

    index = faiss.read_index(str(index_path))
    pids = np.load(pids_path)
    return index, pids


def _load_query_encoder(repo_root: Path) -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """Load tokenizer and query encoder from artifacts/retriever/query_encoder/."""
    encoder_dir = repo_root / "artifacts" / "retriever" / "query_encoder"
    if not encoder_dir.exists():
        raise FileNotFoundError(
            f"Query encoder directory not found: {encoder_dir}. "
            "Train the retriever first so that encoders are saved."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
        model = AutoModel.from_pretrained(encoder_dir)
    except OSError as exc:
        raise FileNotFoundError(
            f"Failed to load query encoder from {encoder_dir}. "
            "Ensure the directory contains a valid HuggingFace model (config.json, model weights)."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _encode_queries(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    batch_size: int,
    max_len: int,
    device: torch.device,
    normalize: bool,
) -> np.ndarray:
    """Encode query texts to float32 embeddings (CLS, optional L2 norm)."""
    if not texts:
        return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)

    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding queries", unit="batch"):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]
            if normalize:
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            all_embeddings.append(cls_emb.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embeddings, axis=0)


def _relevant_pids_per_qid(qrels: pd.DataFrame) -> dict[int, set[int]]:
    """Build qid -> set of pids with relevance > 0."""
    pos = qrels[qrels["relevance"] > 0] if "relevance" in qrels.columns else qrels
    out: dict[int, set[int]] = {}
    for _, row in pos.iterrows():
        qid = int(row["qid"])
        pid = int(row["pid"])
        out.setdefault(qid, set()).add(pid)
    return out


def run_eval(config_path: str) -> None:
    """Load config and data, encode queries, retrieve, compute Recall@K, save results."""
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))

    retriever_cfg = cfg.get("retriever", {})
    faiss_cfg = cfg.get("faiss", {})
    backend = str(faiss_cfg.get("encoder_backend", "hf_biencoder")).strip().lower()
    topk = int(faiss_cfg.get("topk", 100))
    max_len = int(retriever_cfg.get("max_len", 128))
    batch_size = int(retriever_cfg.get("batch_size", 64))
    normalize = bool(retriever_cfg.get("normalize", True))

    repo_root = Path(__file__).resolve().parents[2]

    queries_df, qrels_df = _load_dev_data(processed_dir)
    index, pids = _load_faiss_index(repo_root)
    print(f"FAISS index dim: {index.d}")

    query_texts = queries_df["query"].astype(str).tolist()
    qids = queries_df["qid"].astype("int64").to_numpy()

    if backend == "sentence_transformer":
        st_name = faiss_cfg.get("st_model_name")
        if not st_name:
            raise KeyError("faiss.encoder_backend is 'sentence_transformer' but faiss.st_model_name is not set.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_transformer backend requires the 'sentence-transformers' package. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        model = SentenceTransformer(st_name)
        embeddings = model.encode(
            query_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        print(f"Using backend: sentence_transformer, embeddings.shape: {embeddings.shape}")
    else:
        tokenizer, model, device = _load_query_encoder(repo_root)
        embeddings = _encode_queries(
            tokenizer=tokenizer,
            model=model,
            texts=query_texts,
            batch_size=batch_size,
            max_len=max_len,
            device=device,
            normalize=normalize,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        print(f"Using backend: hf_biencoder, embeddings.shape: {embeddings.shape}")

    if embeddings.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dim ({embeddings.shape[1]}) does not match FAISS index dim ({index.d}). "
            "Rebuild the FAISS index with the same encoder backend (python -m src.indexing.build_faiss --config <config>), "
            "or set faiss.encoder_backend and faiss.st_model_name to match the index."
        )

    k = min(topk, index.ntotal)
    scores, indices = index.search(embeddings, k)

    # Map index positions to pids
    retrieved_pids = pids[indices]  # (n_queries, k)
    relevant_by_qid = _relevant_pids_per_qid(qrels_df)

    ks = [10, 50, 100]
    ks = [x for x in ks if x <= k]
    recalls = {}
    for K in ks:
        hits = 0
        for i, qid in enumerate(qids):
            rel = relevant_by_qid.get(int(qid), set())
            top_k_pids = set(retrieved_pids[i, :K].tolist())
            if rel & top_k_pids:
                hits += 1
        recalls[K] = hits / len(qids) if qids.size else 0.0

    print("Recall@10:", f"{recalls.get(10, 0):.4f}")
    print("Recall@50:", f"{recalls.get(50, 0):.4f}")
    print("Recall@100:", f"{recalls.get(100, 0):.4f}")

    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "retrieval_results.parquet"

    results = pd.DataFrame({
        "qid": qids,
        "retrieved_pids": [retrieved_pids[i].tolist() for i in range(len(qids))],
        "scores": [scores[i].tolist() for i in range(len(qids))],
    })
    results["qid"] = results["qid"].astype("int64")
    results.to_parquet(out_path, index=False)
    print(f"Saved retrieval results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with FAISS and compute Recall@K.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run_eval(args.config)


if __name__ == "__main__":
    main()

"""Hybrid retrieval evaluation: FAISS + SentenceTransformer retrieval, HF bi-encoder re-ranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss  # type: ignore[import]
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.config import get_path, load_config

# Limit FAISS to a single thread for reproducibility / stability.
faiss.omp_set_num_threads(1)
print("FAISS threads: 1")


def _load_dev_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dev queries and qrels. Prefer *_in_corpus variants when present."""
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
            raise ValueError(
                f"dev_queries must have columns 'qid', 'query'; found {queries.columns.tolist()}"
            )
    for col in ("qid", "pid", "relevance"):
        if col not in qrels.columns:
            raise ValueError(
                f"dev_qrels must have columns 'qid', 'pid', 'relevance'; found {qrels.columns.tolist()}"
            )

    return queries, qrels


def _load_faiss_index(repo_root: Path) -> Tuple[faiss.Index, np.ndarray]:
    """Load FAISS index and pid mapping."""
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


def _load_passages_subset(processed_dir: Path, pids: np.ndarray) -> Dict[int, str]:
    """Load passages.parquet and return pid -> passage only for pids present in array."""
    passages_path = processed_dir / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages parquet not found: {passages_path}. "
            "Did you run preprocessing to generate Parquet files?"
        )

    df = pd.read_parquet(passages_path)
    if "pid" not in df.columns or "passage" not in df.columns:
        raise ValueError(
            f"Expected columns 'pid' and 'passage' in {passages_path}, found {df.columns.tolist()}."
        )

    pid_set = set(int(x) for x in pids.tolist())
    df = df[df["pid"].isin(pid_set)]
    if df.empty:
        raise RuntimeError("No passages overlap with pids.npy; nothing to evaluate.")

    mapping = (
        df[["pid", "passage"]]
        .drop_duplicates("pid")
        .set_index("pid")["passage"]
        .astype(str)
        .to_dict()
    )
    # Ensure integer keys
    return {int(k): str(v) for k, v in mapping.items()}


def _load_hf_biencoder(
    repo_root: Path,
) -> Tuple[AutoTokenizer, AutoModel, AutoTokenizer, AutoModel, torch.device]:
    """Load fine-tuned HF query and doc encoders from artifacts/retriever/."""
    query_dir = repo_root / "artifacts" / "retriever" / "query_encoder"
    doc_dir = repo_root / "artifacts" / "retriever" / "doc_encoder"

    if not query_dir.exists():
        raise FileNotFoundError(
            f"Query encoder directory not found: {query_dir}. Train the retriever first."
        )
    if not doc_dir.exists():
        raise FileNotFoundError(
            f"Doc encoder directory not found: {doc_dir}. Train the retriever first."
        )

    try:
        q_tokenizer = AutoTokenizer.from_pretrained(query_dir)
        q_model = AutoModel.from_pretrained(query_dir)
    except OSError as exc:
        raise FileNotFoundError(
            f"Failed to load query encoder from {query_dir}. "
            "Ensure it contains a valid HuggingFace model."
        ) from exc

    try:
        d_tokenizer = AutoTokenizer.from_pretrained(doc_dir)
        d_model = AutoModel.from_pretrained(doc_dir)
    except OSError as exc:
        raise FileNotFoundError(
            f"Failed to load doc encoder from {doc_dir}. "
            "Ensure it contains a valid HuggingFace model."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_model.to(device)
    d_model.to(device)
    q_model.eval()
    d_model.eval()
    return q_tokenizer, q_model, d_tokenizer, d_model, device


def _relevant_pids_per_qid(qrels: pd.DataFrame) -> Dict[int, set[int]]:
    """Build qid -> set of pids with relevance > 0."""
    pos = qrels[qrels["relevance"] > 0] if "relevance" in qrels.columns else qrels
    out: Dict[int, set[int]] = {}
    for _, row in pos.iterrows():
        qid = int(row["qid"])
        pid = int(row["pid"])
        out.setdefault(qid, set()).add(pid)
    return out


def _encode_queries_sentence_transformer(
    model_name: str,
    texts: Sequence[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode queries using SentenceTransformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Hybrid retrieval requires the 'sentence-transformers' package. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if not texts:
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_query_hf(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    text: str,
    max_len: int,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    """Encode a single query to a (1, dim) tensor."""
    with torch.no_grad():
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :]
        if normalize:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)
    return cls_emb  # (1, dim)


def _encode_passages_hf(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: Sequence[str],
    max_len: int,
    device: torch.device,
    normalize: bool,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode passages to a (N, dim) tensor using CLS pooling."""
    if not texts:
        hidden_size = int(model.config.hidden_size)
        return torch.zeros((0, hidden_size), dtype=torch.float32, device=device)

    all_embs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]
            if normalize:
                cls_emb = F.normalize(cls_emb, p=2, dim=-1)
            all_embs.append(cls_emb)
    return torch.cat(all_embs, dim=0)


def run_eval_hybrid(config_path: str) -> None:
    """Run hybrid retrieval evaluation: FAISS+SentenceTransformer then HF bi-encoder re-ranking."""
    cfg = load_config(config_path)

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))

    retriever_cfg: Dict[str, Any] = cfg.get("retriever", {})
    faiss_cfg: Dict[str, Any] = cfg.get("faiss", {})

    max_len = int(retriever_cfg.get("max_len", 128))
    hf_batch_size = int(retriever_cfg.get("batch_size", 64))
    normalize = bool(retriever_cfg.get("normalize", True))

    topk_cfg = int(faiss_cfg.get("topk", 1000))
    candidate_k = min(topk_cfg, 500)
    st_model_name = faiss_cfg.get("st_model_name")
    if not st_model_name:
        raise KeyError("Config must define 'faiss.st_model_name' for hybrid retrieval.")

    repo_root = Path(__file__).resolve().parents[2]

    # Data and index
    queries_df, qrels_df = _load_dev_data(processed_dir)
    index, pids = _load_faiss_index(repo_root)
    candidate_k = min(candidate_k, int(index.ntotal))
    if candidate_k <= 0:
        raise RuntimeError("FAISS index is empty; cannot evaluate hybrid retrieval.")
    print(f"Using candidate_k (per query): {candidate_k}")

    passages_by_pid = _load_passages_subset(processed_dir, pids)

    # Encode queries with SentenceTransformer and search in FAISS
    query_texts = queries_df["query"].astype(str).tolist()
    qids = queries_df["qid"].astype("int64").to_numpy()

    st_embeddings = _encode_queries_sentence_transformer(
        model_name=st_model_name,
        texts=query_texts,
        batch_size=64,
    )
    if st_embeddings.shape[0] != len(query_texts):
        raise RuntimeError(
            f"Mismatch between encoded queries ({st_embeddings.shape[0]}) and input queries ({len(query_texts)})."
        )
    if st_embeddings.shape[1] != index.d:
        raise ValueError(
            f"SentenceTransformer embedding dim ({st_embeddings.shape[1]}) does not match FAISS index dim ({index.d})."
        )

    faiss_scores, faiss_indices = index.search(st_embeddings.astype(np.float32), candidate_k)
    candidate_pids = pids[faiss_indices]  # shape (num_queries, candidate_k)

    # Load HF bi-encoder
    q_tok, q_model, d_tok, d_model, device = _load_hf_biencoder(repo_root)

    # Re-rank with HF bi-encoder and compute metrics
    relevant_by_qid = _relevant_pids_per_qid(qrels_df)
    ks = [10, 50, 100]
    hits_at_k = {K: 0 for K in ks}
    mrr_sum = 0.0
    mrr_k = 100
    num_eval_queries = 0  # queries that have at least one relevant doc

    all_candidate_pids: List[List[int]] = []
    all_hf_scores: List[List[float]] = []

    for i in tqdm(range(len(query_texts)), desc="Re-ranking with HF bi-encoder", unit="query"):
        qid = int(qids[i])
        q_text = query_texts[i]
        cand_pids_row = [int(pid) for pid in candidate_pids[i].tolist()]
        cand_texts = [passages_by_pid.get(pid, "") for pid in cand_pids_row]

        # Encode query once
        q_emb = _encode_query_hf(
            tokenizer=q_tok,
            model=q_model,
            text=q_text,
            max_len=max_len,
            device=device,
            normalize=normalize,
        )  # (1, dim)

        # Encode candidate passages in batches
        d_embs = _encode_passages_hf(
            tokenizer=d_tok,
            model=d_model,
            texts=cand_texts,
            max_len=max_len,
            device=device,
            normalize=normalize,
            batch_size=hf_batch_size,
        )  # (N, dim)

        if d_embs.shape[0] != len(cand_texts):
            raise RuntimeError(
                f"Mismatch between encoded passages ({d_embs.shape[0]}) and candidate texts ({len(cand_texts)})."
            )

        # Dot product scores: (N,)
        with torch.no_grad():
            scores = torch.matmul(d_embs, q_emb.t()).squeeze(-1).cpu().numpy().astype(np.float32)

        order = np.argsort(-scores)
        sorted_pids = [cand_pids_row[j] for j in order]
        sorted_scores = scores[order]

        all_candidate_pids.append(sorted_pids)
        all_hf_scores.append(sorted_scores.tolist())

        rel = relevant_by_qid.get(qid, set())
        if not rel:
            continue  # skip queries without any relevant documents

        num_eval_queries += 1

        # Recall@K
        for K in ks:
            if K > len(sorted_pids):
                continue
            top_k_pids = set(sorted_pids[:K])
            if rel & top_k_pids:
                hits_at_k[K] += 1

        # MRR@mrr_k
        rank = None
        for idx, pid in enumerate(sorted_pids[: min(mrr_k, len(sorted_pids))]):
            if pid in rel:
                rank = idx + 1
                break
        if rank is not None:
            mrr_sum += 1.0 / float(rank)

    if num_eval_queries == 0:
        print("No dev queries have relevant documents; cannot compute Recall or MRR.")
        recall_at_k = {K: 0.0 for K in ks}
        mrr = 0.0
    else:
        recall_at_k = {K: hits_at_k[K] / num_eval_queries for K in ks}
        mrr = mrr_sum / num_eval_queries

    print(f"Number of dev queries: {len(query_texts):,}")
    print(f"Number of eval queries with relevance > 0: {num_eval_queries:,}")
    for K in ks:
        print(f"Hybrid Recall@{K}: {recall_at_k[K]:.4f}")
    print(f"Hybrid MRR@{mrr_k}: {mrr:.4f}")

    # Save per-query results
    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "hybrid_results.parquet"

    results = pd.DataFrame(
        {
            "qid": qids,
            "candidate_pids": all_candidate_pids,
            "hf_scores": all_hf_scores,
        }
    )
    results["qid"] = results["qid"].astype("int64")
    results.to_parquet(out_path, index=False)
    print(f"Saved hybrid retrieval results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval: FAISS+SentenceTransformer retrieval and HF bi-encoder re-ranking.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run_eval_hybrid(args.config)


if __name__ == "__main__":
    main()


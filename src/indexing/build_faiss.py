"""Build a FAISS index over passage embeddings using a trained doc encoder or SentenceTransformer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore[import]
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.config import get_path, load_config, set_seed

# Limit FAISS to a single thread for reproducibility / stability.
faiss.omp_set_num_threads(1)
print("FAISS threads set to 1 in build_faiss.py")


def _load_passages(processed_dir: Path, max_passages: int) -> Tuple[np.ndarray, List[str]]:
    """Load and limit passages from Parquet."""
    passages_path = processed_dir / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages parquet not found: {passages_path}. "
            "Did you run the preprocessing step to generate Parquet files?"
        )

    df = pd.read_parquet(passages_path)
    if "pid" not in df.columns or "passage" not in df.columns:
        raise ValueError(f"Expected columns 'pid' and 'passage' in {passages_path}, found {df.columns.tolist()}.")

    if max_passages > 0:
        df = df.head(max_passages)

    if df.empty:
        raise RuntimeError("No passages available after applying max_passages limit.")

    pids = df["pid"].astype("int64").to_numpy()
    texts = df["passage"].astype(str).tolist()
    return pids, texts


def _load_doc_encoder(repo_root: Path) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """Load tokenizer and doc encoder from artifacts/retriever/doc_encoder/."""
    encoder_dir = repo_root / "artifacts" / "retriever" / "doc_encoder"
    if not encoder_dir.exists():
        raise FileNotFoundError(
            f"Doc encoder directory not found: {encoder_dir}. "
            "Train the retriever first so that encoders are saved."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
        model = AutoModel.from_pretrained(encoder_dir)
    except OSError as exc:  # e.g. missing config or weights
        raise FileNotFoundError(
            f"Failed to load doc encoder from {encoder_dir}. "
            "Ensure the directory contains a valid HuggingFace model (config.json, model weights)."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _encode_passages(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    batch_size: int,
    max_len: int,
    device: torch.device,
    normalize: bool,
) -> np.ndarray:
    """Encode passages into embeddings using CLS pooling."""
    if not texts:
        return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)

    all_embeddings: list[np.ndarray] = []
    start_time = time.perf_counter()

    with torch.no_grad():
        pbar = tqdm(range(0, len(texts), batch_size), desc="Encoding passages", unit="batch")
        for start in pbar:
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            token_embeddings = outputs.last_hidden_state  # (B, L, H)
            cls_embeddings = token_embeddings[:, 0, :]  # (B, H)

            if normalize:
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)

            all_embeddings.append(cls_embeddings.cpu().numpy().astype(np.float32))

    elapsed = time.perf_counter() - start_time
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Encoded {embeddings.shape[0]} passages in {elapsed:.2f}s")
    return embeddings


def _encode_passages_sentence_transformer(
    model_name: str,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Encode passages using SentenceTransformer; returns float32 numpy."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformer backend requires the 'sentence-transformers' package. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if not texts:
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    model = SentenceTransformer(model_name)
    start_time = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - start_time
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"Encoded {embeddings.shape[0]} passages in {elapsed:.2f}s")
    return embeddings


def build_faiss_index(config_path: str) -> None:
    """Entry point for building a FAISS index from passage embeddings."""
    cfg = load_config(config_path)

    # Optional seeding
    retriever_cfg = cfg.get("retriever", {})
    seed = retriever_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Paths and limits
    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))

    limits_cfg = cfg.get("limits", {})
    max_passages = int(limits_cfg.get("max_passages", 0))  # 0 means no explicit cap beyond dataset

    faiss_cfg = cfg.get("faiss", {})
    encoder_backend = str(faiss_cfg.get("encoder_backend", "hf_biencoder")).strip().lower()
    st_model_name = faiss_cfg.get("st_model_name", "")

    retriever_normalize = bool(retriever_cfg.get("normalize", True))
    batch_size = int(retriever_cfg.get("batch_size", 64))
    max_len = int(retriever_cfg.get("max_len", 128))

    # Repo root (assumes this file is at src/indexing/build_faiss.py)
    repo_root = Path(__file__).resolve().parents[2]

    # Load passages
    pids, texts = _load_passages(processed_dir, max_passages)
    n_passages = len(texts)

    if encoder_backend == "sentence_transformer":
        if not st_model_name:
            raise KeyError("faiss.encoder_backend is 'sentence_transformer' but faiss.st_model_name is not set.")
        print(f"Encoder backend: sentence_transformer (model: {st_model_name})")
        encode_start = time.perf_counter()
        embeddings = _encode_passages_sentence_transformer(
            model_name=st_model_name,
            texts=texts,
            batch_size=batch_size,
        )
        encode_time = time.perf_counter() - encode_start
    else:
        if encoder_backend != "hf_biencoder":
            print(f"Unknown encoder_backend '{encoder_backend}', using hf_biencoder.")
        print("Encoder backend: hf_biencoder (artifacts/retriever/doc_encoder)")
        tokenizer, model, device = _load_doc_encoder(repo_root)
        encode_start = time.perf_counter()
        embeddings = _encode_passages(
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            batch_size=batch_size,
            max_len=max_len,
            device=device,
            normalize=retriever_normalize,
        )
        encode_time = time.perf_counter() - encode_start

    if encode_time > 0 and n_passages > 0:
        throughput = n_passages / encode_time
        print(f"Encoding throughput: {throughput:.1f} passages/sec")

    if embeddings.shape[0] != len(pids):
        raise RuntimeError(
            f"Mismatch between number of embeddings ({embeddings.shape[0]}) "
            f"and pids ({len(pids)})."
        )

    dim = embeddings.shape[1]

    # Build FAISS index
    index_start = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    index_time = time.perf_counter() - index_start

    # Save index and pid mapping
    index_dir = repo_root / "artifacts" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    pids_path = index_dir / "pids.npy"

    faiss.write_index(index, str(index_path))
    np.save(pids_path, pids)

    print(f"FAISS index saved to: {index_path}")
    print(f"PIDs mapping saved to: {pids_path}")
    print(f"Count: {embeddings.shape[0]}, Dim: {dim}")
    print(f"Encoding time: {encode_time:.2f}s, Indexing time: {index_time:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from passage embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    build_faiss_index(args.config)


if __name__ == "__main__":
    main()


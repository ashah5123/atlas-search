"""Train a bi-encoder retriever with in-batch negatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.bi_encoder import BiEncoder
from src.utils.config import get_path, load_config, set_seed


class QueryPassageDataset(Dataset):
    """Simple dataset of (query, passage) text pairs."""

    def __init__(self, queries: List[str], passages: List[str]) -> None:
        if len(queries) != len(passages):
            raise ValueError("Queries and passages must have the same length.")
        self._queries = queries
        self._passages = passages

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._queries)

    def __getitem__(self, idx: int) -> Tuple[str, str]:  # type: ignore[override]
        return self._queries[idx], self._passages[idx]


def _load_training_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed Parquet files and validate their presence."""
    train_queries_path = processed_dir / "train_queries.parquet"
    train_qrels_path = processed_dir / "train_qrels.parquet"
    passages_path = processed_dir / "passages.parquet"

    missing = [
        str(p)
        for p in (train_queries_path, train_qrels_path, passages_path)
        if not p.exists()
    ]
    if missing:
        msg = (
            "Missing required processed files. Expected the following Parquet files:\n"
            f"  - {train_queries_path}\n"
            f"  - {train_qrels_path}\n"
            f"  - {passages_path}\n"
            f"Missing: {', '.join(missing)}"
        )
        raise FileNotFoundError(msg)

    train_queries = pd.read_parquet(train_queries_path)
    train_qrels = pd.read_parquet(train_qrels_path)
    passages = pd.read_parquet(passages_path)

    return train_queries, train_qrels, passages


def _build_training_pairs(
    train_queries: pd.DataFrame,
    train_qrels: pd.DataFrame,
    passages: pd.DataFrame,
    seed: int,
) -> QueryPassageDataset:
    """Build (query, passage) text pairs from ALL positive qrels (keep multiple positives per qid)."""
    # All positive qrels: keep every (qid, pid) with relevance > 0
    if "relevance" not in train_qrels.columns:
        raise ValueError("train_qrels must have a 'relevance' column.")
    pairs_df = train_qrels[train_qrels["relevance"] > 0][["qid", "pid"]].copy()

    if pairs_df.empty:
        raise RuntimeError("No positive qrels found in training data (relevance > 0).")

    # Join with train_queries on qid and passages on pid to get text columns
    merged = pairs_df.merge(train_queries[["qid", "query"]], on="qid", how="inner")
    merged = merged.merge(passages[["pid", "passage"]], on="pid", how="inner")

    if merged.empty:
        raise RuntimeError("Training pairs are empty after joining with queries and passages.")

    # Shuffle with configured seed for reproducibility
    merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)

    queries = merged["query"].astype(str).tolist()
    ps = merged["passage"].astype(str).tolist()

    return QueryPassageDataset(queries, ps)


def train_retriever(config_path: str) -> None:
    """Main training routine for the bi-encoder retriever."""
    cfg = load_config(config_path)

    # Paths
    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    artifacts_dir_str = get_path(cfg, "paths", "artifacts_dir")
    if processed_dir_str is None or artifacts_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir' and 'paths.artifacts_dir'.")

    processed_dir = Path(str(processed_dir_str))
    artifacts_dir = Path(str(artifacts_dir_str))

    # Retriever config
    retriever_cfg = cfg.get("retriever", {})
    model_name = retriever_cfg.get("model_name")
    if not model_name:
        raise KeyError("Config must define 'retriever.model_name'.")

    batch_size = int(retriever_cfg.get("batch_size", 32))
    max_len = int(retriever_cfg.get("max_len", 128))
    lr = float(retriever_cfg.get("lr", 2e-5))
    epochs = int(retriever_cfg.get("epochs", 1))
    normalize = bool(retriever_cfg.get("normalize", True))
    seed = int(retriever_cfg.get("seed", 42))

    set_seed(seed)

    # Load data
    train_queries, train_qrels, passages = _load_training_data(processed_dir)
    dataset = _build_training_pairs(train_queries, train_qrels, passages, seed=seed)

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    num_pairs = len(dataset)
    batches_per_epoch = (num_pairs + batch_size - 1) // batch_size
    print(f"Number of training pairs: {num_pairs:,}")
    print(f"Expected batches per epoch: {batches_per_epoch:,}")

    # Model
    bi_encoder = BiEncoder(model_name=model_name, normalize=normalize, device=None)
    tokenizer = bi_encoder.tokenizer
    device = bi_encoder.device

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        list(bi_encoder.query_encoder.parameters()) + list(bi_encoder.doc_encoder.parameters()),
        lr=lr,
    )
    criterion = torch.nn.CrossEntropyLoss()

    def collate_fn(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        qs, ps = zip(*batch)
        return list(qs), list(ps)

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
        generator=generator,
    )

    bi_encoder.query_encoder.train()
    bi_encoder.doc_encoder.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for queries, passages_text in pbar:
            optimizer.zero_grad()

            # Tokenize
            q_tokens = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            p_tokens = tokenizer(
                passages_text,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            # Forward
            q_outputs = bi_encoder.query_encoder(**q_tokens)
            p_outputs = bi_encoder.doc_encoder(**p_tokens)

            q_cls = q_outputs.last_hidden_state[:, 0, :]
            p_cls = p_outputs.last_hidden_state[:, 0, :]

            if normalize:
                q_cls = F.normalize(q_cls, p=2, dim=-1)
                p_cls = F.normalize(p_cls, p=2, dim=-1)

            # In-batch negatives similarity matrix
            logits = q_cls @ p_cls.t()
            targets = torch.arange(logits.size(0), device=device)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(bi_encoder.query_encoder.parameters()) + list(bi_encoder.doc_encoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            batch_loss = float(loss.item())
            epoch_loss += batch_loss
            num_batches += 1

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch} finished. Average loss: {avg_loss:.4f}")

    # Save encoders and tokenizer
    retriever_dir = artifacts_dir / "retriever"
    query_dir = retriever_dir / "query_encoder"
    doc_dir = retriever_dir / "doc_encoder"
    query_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(query_dir)
    bi_encoder.query_encoder.save_pretrained(query_dir)

    tokenizer.save_pretrained(doc_dir)
    bi_encoder.doc_encoder.save_pretrained(doc_dir)

    print(f"Saved query encoder to: {query_dir}")
    print(f"Saved doc encoder to: {doc_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bi-encoder retriever with in-batch negatives.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    train_retriever(args.config)


if __name__ == "__main__":
    main()


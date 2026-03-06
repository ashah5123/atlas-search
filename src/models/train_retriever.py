"""Train a bi-encoder retriever with in-batch negatives."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


class QueryPosDataset(Dataset):
    """Dataset of (qid, query, positive_pid) triples."""

    def __init__(self, qids: List[int], queries: List[str], pos_pids: List[int]) -> None:
        if not (len(qids) == len(queries) == len(pos_pids)):
            raise ValueError("qids, queries, and pos_pids must have the same length.")
        self._qids = qids
        self._queries = queries
        self._pos_pids = pos_pids

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._qids)

    def __getitem__(self, idx: int) -> Tuple[int, str, int]:  # type: ignore[override]
        return self._qids[idx], self._queries[idx], self._pos_pids[idx]


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


def _load_hard_negs(processed_dir: Path) -> Dict[int, List[int]]:
    """Load train_hard_negs.parquet if present; return qid -> list of neg_pid."""
    path = processed_dir / "train_hard_negs.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "qid" not in df.columns or "neg_pid" not in df.columns:
        raise ValueError(
            f"train_hard_negs.parquet must have 'qid', 'neg_pid'; found {df.columns.tolist()}"
        )
    hard_negs: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        qid = int(row["qid"])
        pid = int(row["neg_pid"])
        hard_negs.setdefault(qid, []).append(pid)
    return hard_negs


def _build_training_pairs(
    train_queries: pd.DataFrame,
    train_qrels: pd.DataFrame,
    passages: pd.DataFrame,
    seed: int,
    max_train_pairs: int | None = None,
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
    if max_train_pairs is not None:
        merged = merged.head(max_train_pairs)

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
    neg_per_query = int(retriever_cfg.get("neg_per_query", 5))
    max_train_pairs = retriever_cfg.get("max_train_pairs")  # None or int

    set_seed(seed)

    # Load data
    train_queries, train_qrels, passages = _load_training_data(processed_dir)
    hard_negs_by_qid = _load_hard_negs(processed_dir)

    use_hard_negs = bool(hard_negs_by_qid)
    if use_hard_negs:
        # Build positive pairs only for qids that have mined negatives
        if "relevance" not in train_qrels.columns:
            raise ValueError("train_qrels must have a 'relevance' column.")
        pairs_df = train_qrels[train_qrels["relevance"] > 0][["qid", "pid"]].copy()
        if pairs_df.empty:
            raise RuntimeError("No positive qrels found in training data (relevance > 0).")
        pairs_df = pairs_df[pairs_df["qid"].isin(hard_negs_by_qid.keys())]
        if pairs_df.empty:
            raise RuntimeError("No positive pairs have corresponding hard negatives; cannot train.")
        merged = pairs_df.merge(train_queries[["qid", "query"]], on="qid", how="inner")
        if merged.empty:
            raise RuntimeError("Training pairs are empty after joining with queries.")
        merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
        if max_train_pairs is not None:
            merged = merged.head(max_train_pairs)
        qids = merged["qid"].astype("int64").tolist()
        queries_list = merged["query"].astype(str).tolist()
        pos_pids = merged["pid"].astype("int64").tolist()
        dataset_hard = QueryPosDataset(qids, queries_list, pos_pids)
        if len(dataset_hard) == 0:
            raise RuntimeError("Training dataset is empty after filtering for hard negatives.")
        num_pairs = len(dataset_hard)
        batches_per_epoch = (num_pairs + batch_size - 1) // batch_size
        print(f"Using mined hard negatives.")
        print(f"Final number of training pairs used: {num_pairs:,}")
        print(f"neg_per_query used: {neg_per_query}")
        print(f"Expected batches per epoch: {batches_per_epoch:,}")
    else:
        dataset = _build_training_pairs(
            train_queries, train_qrels, passages, seed=seed, max_train_pairs=max_train_pairs
        )
        if len(dataset) == 0:
            raise RuntimeError("Training dataset is empty.")
        num_pairs = len(dataset)
        batches_per_epoch = (num_pairs + batch_size - 1) // batch_size
        print(f"Final number of training pairs used: {num_pairs:,}")
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

    generator = torch.Generator()
    generator.manual_seed(seed)

    bi_encoder.query_encoder.train()
    bi_encoder.doc_encoder.train()

    if use_hard_negs:
        # Pre-build pid -> passage mapping and global pid list for fallback
        passage_by_pid = (
            passages[["pid", "passage"]]
            .drop_duplicates("pid")
            .set_index("pid")["passage"]
            .astype(str)
            .to_dict()
        )
        passage_by_pid = {int(k): str(v) for k, v in passage_by_pid.items()}
        all_pids: List[int] = list(passage_by_pid.keys())
        if not all_pids:
            raise RuntimeError("No passages available for training.")

        def collate_hard(batch: List[Tuple[int, str, int]]) -> Tuple[List[int], List[str], List[int]]:
            qids_b, qs_b, pos_pids_b = zip(*batch)
            return list(qids_b), list(qs_b), list(pos_pids_b)

        dataloader = DataLoader(
            dataset_hard,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_hard,
            drop_last=False,
            generator=generator,
        )

        group_size = 1 + neg_per_query

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
            for qids_batch, queries_batch, pos_pids_batch in pbar:
                optimizer.zero_grad()

                B = len(queries_batch)

                # Encode queries -> Q (B, dim)
                q_tokens = tokenizer(
                    queries_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                q_outputs = bi_encoder.query_encoder(**q_tokens)
                q_cls = q_outputs.last_hidden_state[:, 0, :]

                # Build doc_texts: for each i in batch order, [pos] + exactly neg_per_query negs
                doc_texts: List[str] = []
                for i in range(B):
                    qid = int(qids_batch[i])
                    pos_pid = int(pos_pids_batch[i])
                    pos_text = passage_by_pid.get(pos_pid, "")
                    doc_texts.append(pos_text)

                    neg_candidates = hard_negs_by_qid.get(qid, [])
                    if len(neg_candidates) >= neg_per_query:
                        neg_sample = random.sample(neg_candidates, neg_per_query)
                    elif len(neg_candidates) > 0:
                        neg_sample = random.choices(neg_candidates, k=neg_per_query)
                    else:
                        pool = [p for p in all_pids if p != pos_pid]
                        if not pool:
                            pool = [pos_pid]
                        neg_sample = random.choices(pool, k=neg_per_query)

                    for neg_pid in neg_sample:
                        neg_text = passage_by_pid.get(int(neg_pid), "")
                        doc_texts.append(neg_text)

                assert len(doc_texts) == B * (1 + neg_per_query), (
                    f"doc_texts length {len(doc_texts)} != B*(1+neg_per_query) = {B * (1 + neg_per_query)}"
                )

                d_tokens = tokenizer(
                    doc_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                d_outputs = bi_encoder.doc_encoder(**d_tokens)
                d_cls = d_outputs.last_hidden_state[:, 0, :]

                if normalize:
                    q_cls_norm = F.normalize(q_cls, p=2, dim=-1)
                    d_cls_norm = F.normalize(d_cls, p=2, dim=-1)
                else:
                    q_cls_norm = q_cls
                    d_cls_norm = d_cls

                logits = q_cls_norm @ d_cls_norm.t()

                assert logits.shape[1] == B * (1 + neg_per_query), (
                    f"logits.shape[1] {logits.shape[1]} != B*(1+neg_per_query) {B * (1 + neg_per_query)}"
                )
                targets = torch.arange(B, device=logits.device, dtype=torch.long) * (1 + neg_per_query)
                assert targets.max().item() < logits.shape[1], (
                    f"targets.max() {targets.max().item()} >= logits.shape[1] {logits.shape[1]}"
                )

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
    else:
        def collate_fn(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
            qs, ps = zip(*batch)
            return list(qs), list(ps)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
            generator=generator,
        )

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
            for queries_batch, passages_text in pbar:
                optimizer.zero_grad()

                # Tokenize
                q_tokens = tokenizer(
                    queries_batch,
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


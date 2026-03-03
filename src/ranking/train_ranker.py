"""Train a LightGBM learning-to-rank model on the LTR dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config


def _load_ltr_data(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dev_ltr.parquet and dev_groups.npy from data/processed/ltr/."""
    ltr_dir = repo_root / "data" / "processed" / "ltr"
    parquet_path = ltr_dir / "dev_ltr.parquet"
    groups_path = ltr_dir / "dev_groups.npy"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"LTR parquet not found: {parquet_path}. "
            "Run src.ranking.build_ltr_dataset first."
        )
    if not groups_path.exists():
        raise FileNotFoundError(
            f"LTR groups not found: {groups_path}. "
            "Run src.ranking.build_ltr_dataset first."
        )

    df = pd.read_parquet(parquet_path)
    groups = np.load(groups_path)
    return df, groups


def train_ranker(config_path: str) -> None:
    cfg = load_config(config_path)

    repo_root = Path(__file__).resolve().parents[2]
    df, dev_groups = _load_ltr_data(repo_root)

    # Features = all numeric columns except qid, pid, label
    exclude = {"qid", "pid", "label"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [c for c in numeric_cols if c not in exclude]
    if not feature_names:
        raise ValueError(
            "No feature columns found. Need at least one numeric column other than qid, pid, label."
        )

    # Unique qids in order of appearance (matches dev_groups order)
    qid_order = df["qid"].unique()
    n_queries = len(qid_order)
    if n_queries != len(dev_groups):
        raise RuntimeError(
            f"Group count mismatch: {len(dev_groups)} groups vs {n_queries} unique qids in dev_ltr."
        )

    # 80% train, 20% valid by query
    n_train = int(0.8 * n_queries)
    train_qids = set(qid_order[:n_train])
    valid_qids = set(qid_order[n_train:])

    train_df = df[df["qid"].isin(train_qids)].copy()
    valid_df = df[df["qid"].isin(valid_qids)].copy()

    train_group = np.array(
        [len(block) for _, block in train_df.groupby("qid", sort=False)],
        dtype=np.int32,
    )
    valid_group = np.array(
        [len(block) for _, block in valid_df.groupby("qid", sort=False)],
        dtype=np.int32,
    )

    X_train = train_df[feature_names]
    y_train = train_df["label"].values
    X_valid = valid_df[feature_names]
    y_valid = valid_df["label"].values

    ranker_cfg = cfg.get("ranker", {})
    objective = str(ranker_cfg.get("objective", "lambdarank"))
    num_leaves = int(ranker_cfg.get("num_leaves", 127))
    learning_rate = float(ranker_cfg.get("learning_rate", 0.05))
    n_estimators = int(ranker_cfg.get("n_estimators", 200))
    eval_at_val = int(ranker_cfg.get("eval_at", 10))
    eval_at = [eval_at_val]

    model = lgb.LGBMRanker(
        objective=objective,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_group],
        eval_at=eval_at,
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    # Best validation NDCG@eval_at
    evals = model.evals_result_
    valid_key = "valid_0"
    if valid_key in evals:
        for key, values in evals[valid_key].items():
            if "ndcg" in key.lower():
                best_ndcg = max(values) if values else 0.0
                print(f"Best validation {key}: {best_ndcg:.4f}")
                break
        else:
            # Fallback: last value of first metric
            first_metric = next(iter(evals[valid_key].values()), [])
            best_ndcg = max(first_metric) if first_metric else 0.0
            print(f"Best validation NDCG@{eval_at_val}: {best_ndcg:.4f}")
    else:
        print("No validation metrics in evals_result_.")

    out_dir = repo_root / "artifacts" / "ranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "lgbm_ranker.txt"
    names_path = out_dir / "feature_names.json"

    model.booster_.save_model(str(model_path))
    with open(names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved feature names to {names_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker on LTR dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_ranker(args.config)


if __name__ == "__main__":
    main()

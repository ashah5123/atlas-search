"""FastAPI search API: encode query, FAISS retrieval, LightGBM rerank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss  # type: ignore[import]
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from src.utils.config import get_path, load_config


# --- Pydantic models ---


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    topk: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    candidates_k: int = Field(default=100, ge=1, le=500, description="Number of candidates to retrieve before rerank")


class SearchResultItem(BaseModel):
    pid: int
    rerank_score: float
    dense_score: float
    passage: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


# --- App state ---


@dataclass
class AppState:
    cfg: dict[str, Any]
    backend: str
    index: Any  # faiss.Index
    pids: np.ndarray
    pid_to_passage: dict[int, str]
    ranker: Any  # lightgbm.Booster
    feature_names: list[str]
    st_model: Optional[Any] = None
    hf_tokenizer: Optional[Any] = None
    hf_model: Optional[Any] = None
    hf_device: Optional[torch.device] = None
    max_len: int = 128
    normalize: bool = True


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _load_assets(config_path: str) -> AppState:
    """Load FAISS, passages (indexed only), ranker, and query encoder from config."""
    cfg = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[2]

    processed_dir_str = get_path(cfg, "paths", "processed_dir")
    if processed_dir_str is None:
        raise KeyError("Config must define 'paths.processed_dir'.")
    processed_dir = Path(str(processed_dir_str))

    faiss_cfg = cfg.get("faiss", {})
    retriever_cfg = cfg.get("retriever", {})

    # FAISS index and pids
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
    index_pid_set = set(pids.tolist())

    # Passages: only for pids in index
    passages_path = processed_dir / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages not found: {passages_path}. Run preprocessing first."
        )
    passages_df = pd.read_parquet(passages_path)
    if "pid" not in passages_df.columns or "passage" not in passages_df.columns:
        raise ValueError(
            f"passages.parquet must have 'pid', 'passage'; found {passages_df.columns.tolist()}"
        )
    passages_sub = passages_df[passages_df["pid"].isin(index_pid_set)]
    passage_by_pid = (
        passages_sub[["pid", "passage"]]
        .drop_duplicates("pid")
        .set_index("pid")["passage"]
        .astype(str)
        .to_dict()
    )
    passages_loaded_count = len(passage_by_pid)

    # Ranker
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
    ranker = lgb.Booster(model_file=str(ranker_path))
    with open(names_path) as f:
        feature_names = json.load(f)

    # Query encoder
    backend = str(faiss_cfg.get("encoder_backend", "hf_biencoder")).strip().lower()
    max_len = int(retriever_cfg.get("max_len", 128))
    normalize = bool(retriever_cfg.get("normalize", True))

    st_model: Optional[Any] = None
    hf_tokenizer: Optional[Any] = None
    hf_model: Optional[Any] = None
    hf_device: Optional[torch.device] = None

    if backend == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_transformer backend requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from exc
        st_name = faiss_cfg.get("st_model_name")
        if not st_name:
            raise KeyError("faiss.encoder_backend is 'sentence_transformer' but faiss.st_model_name is not set.")
        st_model = SentenceTransformer(st_name)
    else:
        encoder_dir = repo_root / "artifacts" / "retriever" / "query_encoder"
        if not encoder_dir.exists():
            raise FileNotFoundError(
                f"Query encoder not found: {encoder_dir}. Train the retriever first."
            )
        hf_tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
        hf_model = AutoModel.from_pretrained(encoder_dir)
        hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.to(hf_device)
        hf_model.eval()

    return AppState(
        cfg=cfg,
        backend=backend,
        index=index,
        pids=pids,
        pid_to_passage=passage_by_pid,
        ranker=ranker,
        feature_names=feature_names,
        st_model=st_model,
        hf_tokenizer=hf_tokenizer,
        hf_model=hf_model,
        hf_device=hf_device,
        max_len=max_len,
        normalize=normalize,
    )


def _encode_query(state: AppState, query: str) -> np.ndarray:
    """Return (1, dim) float32 query embedding."""
    if state.backend == "sentence_transformer":
        if state.st_model is None:
            raise RuntimeError("SentenceTransformer model is not loaded.")
        emb = state.st_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)
    else:
        if state.hf_tokenizer is None or state.hf_model is None or state.hf_device is None:
            raise RuntimeError("HF query encoder is not loaded.")
        tokenizer = state.hf_tokenizer
        model = state.hf_model
        device = state.hf_device
        with torch.no_grad():
            enc = tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=state.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]
            if state.normalize:
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            return cls_emb.cpu().numpy().astype(np.float32)


def search(state: AppState, query: str, topk: int, candidates_k: int) -> List[Dict[str, Any]]:
    """Run encode -> FAISS -> features -> ranker -> top topk."""
    # 1. Encode query
    q_emb = _encode_query(state, query)
    # 2. FAISS search
    k = min(candidates_k, state.index.ntotal)
    scores, indices = state.index.search(q_emb, k)
    scores = scores[0]
    indices = indices[0]
    cand_pids = state.pids[indices].tolist()
    cand_dense = scores.tolist()

    if not cand_pids:
        return []

    # Resolve passages (skip missing)
    cand_passages: List[str] = []
    valid_pids: List[int] = []
    valid_dense: List[float] = []
    for pid, d in zip(cand_pids, cand_dense):
        pid_int = int(pid)
        text = state.pid_to_passage.get(pid_int)
        if text is None:
            continue
        cand_passages.append(text)
        valid_pids.append(pid_int)
        valid_dense.append(float(d))

    if not valid_pids:
        return []

    query_tokens = _tokenize(query)
    q_len = len(query_tokens)
    docs_tokens = [_tokenize(p) for p in cand_passages]

    # 3. BM25 over candidates
    bm25 = BM25Okapi(docs_tokens)
    bm25_scores = bm25.get_scores(query_tokens).tolist()

    # 4. Build feature rows
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
    df = pd.DataFrame(rows)[state.feature_names]

    # 5. Ranker predict and sort
    pred = state.ranker.predict(df)
    order = np.argsort(-np.asarray(pred))
    # 6. Top topk
    n = min(topk, len(order))
    result_items: List[Dict[str, Any]] = []
    for j in range(n):
        idx = order[j]
        result_items.append({
            "pid": valid_pids[idx],
            "rerank_score": float(pred[idx]),
            "dense_score": valid_dense[idx],
            "passage": cand_passages[idx],
        })
    return result_items


# --- FastAPI app ---

CONFIG_PATH = "configs/config.yaml"


def create_app() -> FastAPI:
    app = FastAPI(title="Atlas Search API", description="Search with FAISS + LightGBM rerank")

    @app.get("/health")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def get_info() -> dict[str, Any]:
        state: AppState = getattr(app.state, "search_state", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Search not initialized")
        backend = (
            state.cfg.get("faiss", {}).get("encoder_backend")
            if isinstance(state.cfg, dict)
            else None
        )
        return {
            "backend": backend,
            "index_dim": int(state.index.d),
            "index_size": int(state.index.ntotal),
            "feature_names_count": int(len(state.feature_names)),
        }

    @app.on_event("startup")
    def startup() -> None:
        config_path = Path(__file__).resolve().parents[2] / CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        state = _load_assets(str(config_path))
        app.state.search_state = state
        backend = state.backend
        dim = state.index.d
        ntotal = state.index.ntotal
        n_passages = len(state.pid_to_passage)
        print(f"Backend: {backend}")
        print(f"FAISS index dim: {dim}, ntotal: {ntotal}")
        print(f"Passages loaded (indexed only): {n_passages}")

    @app.post("/search", response_model=SearchResponse)
    def post_search(request: SearchRequest) -> SearchResponse:
        state: AppState = getattr(app.state, "search_state", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Search not initialized")
        topk = min(50, request.topk)
        candidates_k = min(500, request.candidates_k)
        try:
            results = search(state, request.query, topk=topk, candidates_k=candidates_k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return SearchResponse(query=request.query, results=[SearchResultItem(**r) for r in results])

    return app


app = create_app()

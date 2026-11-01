"""FastAPI backend: ANN + Fuzzy student performance prediction service.

Run:  uvicorn backend.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from . import database as db
from .ann_model import ANNPredictor
from .config import FEATURE_META, FEATURES, MODEL_PATH, TARGET
from .data_generator import generate_dataset
from .fuzzy_system import FuzzyPerformanceSystem
from .hybrid import HybridEngine
from .schemas import BatchRequest, StudentInput, TrainRequest

ann = ANNPredictor()
fuzzy = FuzzyPerformanceSystem()
engine = HybridEngine(ann, fuzzy)
train_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    loaded = False
    if MODEL_PATH.exists():
        try:
            ann.load()
            loaded = True
            print(f"[startup] Loaded ANN model from {MODEL_PATH}")
        except Exception as exc:  # corrupted / incompatible file
            print(f"[startup] Could not load saved model ({exc}); retraining.")
    if not loaded:
        print("[startup] Training ANN on synthetic data (one-off, a few seconds)...")
        with train_lock:
            ann.train(generate_dataset(3000))
            ann.save()
        print("[startup] Model ready.")
    yield


app = FastAPI(title="Student Performance Prediction (ANN + Fuzzy)", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": ann.is_ready}


@app.get("/model/info")
def model_info():
    return {
        "model_ready": ann.is_ready,
        "features": FEATURES,
        "target": TARGET,
        "feature_meta": FEATURE_META,
        "model": ann.meta,
    }


@app.post("/predict")
def predict(student: StudentInput):
    res = engine.predict_many([student.model_dump()], detail=True)[0]
    db.log_prediction(res)
    return res


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {"count": len(req.students),
            "results": engine.predict_many([s.model_dump() for s in req.students])}


@app.get("/sample-data")
def sample_data(n: int = 25):
    n = int(np.clip(n, 1, 1000))
    df = generate_dataset(n, seed=int(np.random.randint(0, 10_000)))
    df.insert(0, "student_name", [f"Student {i + 1}" for i in range(n)])
    return {"records": df.to_dict(orient="records")}


@app.post("/train")
def train(req: TrainRequest):
    if req.source == "custom":
        if not req.records:
            raise HTTPException(422, "source='custom' requires 'records'.")
        df = pd.DataFrame(req.records)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise HTTPException(422, f"Missing columns in uploaded data: {missing}")
        df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
        df[TARGET] = df[TARGET].clip(0, 100)
        if len(df) < 100:
            raise HTTPException(422, f"Need at least 100 valid rows, got {len(df)}.")
    else:
        df = generate_dataset(req.n_samples, seed=req.seed)

    if not train_lock.acquire(blocking=False):
        raise HTTPException(409, "A training job is already running.")
    try:
        meta = ann.train(df, hidden_layers=req.hidden_layers, max_iter=req.max_iter, seed=req.seed)
        ann.save()
    finally:
        train_lock.release()
    return {"message": "Model trained and saved.", "model": meta}


@app.get("/history")
def history(limit: int = 200):
    return {"records": db.fetch_history(limit)}


@app.delete("/history")
def clear_history():
    return {"deleted": db.clear_history()}

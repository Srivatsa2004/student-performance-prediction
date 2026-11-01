"""Artificial Neural Network (multi-layer perceptron) for score prediction."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_PATH, TARGET


class ANNPredictor:
    """Standardiser + MLPRegressor pipeline with training metadata."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.meta: dict = {}

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    # ------------------------------------------------------------------ train
    def train(
        self,
        df: pd.DataFrame,
        hidden_layers: Sequence[int] = (64, 32),
        max_iter: int = 600,
        seed: int = 42,
        test_size: float = 0.2,
    ) -> dict:
        df = df[FEATURES + [TARGET]].astype(float).dropna()
        X, y = df[FEATURES], df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(int(h) for h in hidden_layers),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.003,
                        alpha=1e-3,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=25,
                        max_iter=int(max_iter),
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)

        pred = np.clip(pipe.predict(X_te), 0, 100)
        metrics = {
            "mae": float(mean_absolute_error(y_te, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "r2": float(r2_score(y_te, pred)),
        }
        mlp: MLPRegressor = pipe.named_steps["mlp"]
        self.pipeline = pipe
        self.meta = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_samples": int(len(df)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "hidden_layers": [int(h) for h in hidden_layers],
            "epochs_run": int(mlp.n_iter_),
            "metrics": metrics,
        }
        return self.meta

    # --------------------------------------------------------------- inference
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("ANN model is not trained yet.")
        return np.clip(self.pipeline.predict(rows[FEATURES].astype(float)), 0, 100)

    # ------------------------------------------------------------ persistence
    def save(self, path=MODEL_PATH) -> None:
        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path)

    def load(self, path=MODEL_PATH) -> None:
        blob = joblib.load(path)
        self.pipeline, self.meta = blob["pipeline"], blob["meta"]

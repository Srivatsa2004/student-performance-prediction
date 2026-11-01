"""Synthetic student dataset generator.

Used to bootstrap the ANN when no real dataset is supplied. The relationship
between the inputs and the final score is deliberately non-linear
(diminishing returns for study time, an optimal sleep window, an attendance
penalty) so the neural network has something meaningful to learn.
"""
import numpy as np
import pandas as pd

from .config import FEATURES, TARGET


def generate_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples
    ability = rng.normal(0, 1, n)  # latent "academic ability"

    previous = np.clip(65 + 13 * ability + rng.normal(0, 6, n), 10, 100)
    attendance = np.clip(80 + 8 * ability + rng.normal(0, 10, n), 25, 100)
    study = np.clip(12 + 4 * ability + rng.normal(0, 5, n), 0, 40)
    assignment = np.clip(66 + 10 * ability + rng.normal(0, 8, n), 5, 100)
    participation = np.clip(6 + 1.2 * ability + rng.normal(0, 1.5, n), 0, 10)
    sleep = np.clip(rng.normal(7, 1.1, n), 3, 12)

    score = (
        3
        + 0.30 * previous
        + 0.20 * assignment
        + 0.14 * attendance
        + 22 * (1 - np.exp(-study / 10))           # diminishing returns
        + 0.9 * participation
        - np.minimum(0.8 * (sleep - 7.5) ** 2, 8)   # optimal sleep window
        - np.where(attendance < 60, 0.3 * (60 - attendance), 0)
        + rng.normal(0, 2.5, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame(
        {
            "attendance": attendance,
            "study_hours": study,
            "previous_score": previous,
            "assignment_score": assignment,
            "participation": participation,
            "sleep_hours": sleep,
            TARGET: score,
        }
    )
    return df[FEATURES + [TARGET]].round(2)

"""Central configuration shared by every backend module."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ann_model.joblib"
DB_PATH = DATA_DIR / "history.db"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_score",
    "participation",
    "sleep_hours",
]
TARGET = "final_score"

# Share of the final score that comes from the ANN (the rest comes from the fuzzy system).
# Fixed internally so end users never have to think about it.
ANN_WEIGHT = 0.6

# Used by the frontend to build sliders and by the API to validate ranges.
FEATURE_META = {
    "attendance":       {"label": "Attendance (%)",             "min": 0.0, "max": 100.0, "default": 80.0, "step": 1.0},
    "study_hours":      {"label": "Study hours per week",       "min": 0.0, "max": 40.0,  "default": 12.0, "step": 0.5},
    "previous_score":   {"label": "Previous exam score (%)",    "min": 0.0, "max": 100.0, "default": 65.0, "step": 1.0},
    "assignment_score": {"label": "Assignment average (%)",     "min": 0.0, "max": 100.0, "default": 68.0, "step": 1.0},
    "participation":    {"label": "Class participation (0-10)", "min": 0.0, "max": 10.0,  "default": 6.0,  "step": 0.5},
    "sleep_hours":      {"label": "Sleep hours per night",      "min": 3.0, "max": 12.0,  "default": 7.0,  "step": 0.5},
}

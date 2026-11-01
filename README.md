# 🎓 Student Performance Prediction: ANN + Fuzzy Logic

A hybrid, **neuro-fuzzy** system that predicts a student's final score, grade and
risk level, and explains *why*.

```
┌───────────────────────┐   REST/JSON   ┌───────────────────────────────────────────┐
│  Streamlit frontend   │ ────────────► │  FastAPI backend                          │
│  frontend/app.py      │ ◄──────────── │                                           │
└───────────────────────┘               │  6 raw features ─► ANN (MLP) ─► ANN score │
                                        │        │                    │             │
                                        │        └─► Fuzzy inference ◄┘             │
                                        │            (29 Mamdani rules)             │
                                        │  Final = 0.6·ANN + 0.4·Fuzzy ─► grade/risk │
                                        │  SQLite history · joblib model storage    │
                                        └───────────────────────────────────────────┘
```

## Quick start

```bash
cd student_performance_system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

`run.py` starts the backend (port 8000), waits until it is ready, then starts the
Streamlit UI at **http://localhost:8501**. The first launch trains the ANN on
synthetic data (a few seconds) and saves it to `models/`.

Prefer two terminals?

```bash
uvicorn backend.main:app --port 8000     # terminal 1  (API docs: http://127.0.0.1:8000/docs)
streamlit run frontend/app.py            # terminal 2
```

Run the tests with `pytest`.

## How the hybrid works

| Stage | What it does |
|-------|--------------|
| **ANN** | `StandardScaler → MLPRegressor(64,32)` learns the non-linear mapping from attendance, study hours, previous score, assignment average, participation and sleep to the final score (0-100). |
| **Fuzzy system** | Mamdani inference (min/max, centroid defuzzification). Inputs: ANN score, attendance, study hours, assignments, participation, each with 3 linguistic terms. Output: performance index with 5 terms (poor → excellent). 29 weighted IF-THEN rules. |
| **Fusion** | `final = 0.6·ANN + 0.4·Fuzzy` (the blend is fixed internally; change `ANN_WEIGHT` in `backend/config.py` if you want). Mapped to grade A-F, risk (Low/Medium/High), pass/at-risk, plus personalised recommendations. |

The ANN provides accuracy; the fuzzy layer provides interpretability: you can see
which rules fired and how strongly.

## App pages

- **Predict**: single student, gauge, ANN vs fuzzy scores, recommendations, rule-level explanation.
- **Batch prediction**: upload a CSV, get distributions, risk counts and a downloadable results file.
- **Model accuracy**: R², average error (MAE) and RMSE measured on unseen test students.
- **Train model**: retrain on synthetic data or **your own CSV** with a custom architecture.
- **History**: every single prediction is stored in SQLite; download or clear it.

## Using your own data

Train page → *My own CSV*. Required columns (`final_score` is the 0-100 target):

`attendance, study_hours, previous_score, assignment_score, participation, sleep_hours, final_score`

Ranges: attendance/previous/assignment/final 0-100, study_hours 0-40, participation 0-10, sleep 3-12.
To use different features, edit `FEATURES` / `FEATURE_META` in `backend/config.py`
(and the fuzzy variables in `backend/fuzzy_system.py` if you want them in the rules).

## API summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | liveness + model status |
| GET | `/model/info` | metrics, importances, feature ranges |
| POST | `/predict` | single prediction with fuzzy details (logged) |
| POST | `/predict/batch` | up to 5000 students |
| POST | `/train` | retrain (synthetic or custom records) |
| GET/DELETE | `/history` | prediction log |

## Configuration

Environment variables: `API_HOST`, `API_PORT`, `UI_PORT` (read by `run.py`) and `API_URL` (read by the frontend).

## Project layout

```
backend/   config.py · data_generator.py · ann_model.py · fuzzy_system.py
           hybrid.py · database.py · schemas.py · main.py (FastAPI)
frontend/  app.py (Streamlit)
tests/     test_system.py
run.py     one-command launcher
```

> The bundled dataset is **synthetic**, so the reported accuracy shows the pipeline
> works, not real-world validity. Train on real institutional data before relying on predictions.

"""Hybrid ANN + Fuzzy engine.

1. The ANN predicts a numeric score from all six raw features.
2. The ANN score (plus behavioural inputs) is fed into the fuzzy system, which
   applies human-readable linguistic rules and yields a performance index.
3. Final score = w * ANN + (1 - w) * Fuzzy, mapped to a grade and risk level.
"""
from __future__ import annotations

import pandas as pd

from .ann_model import ANNPredictor
from .config import ANN_WEIGHT, FEATURES
from .fuzzy_system import FuzzyPerformanceSystem

GRADE_BANDS = [
    (85, "A", "Excellent"),
    (70, "B", "Good"),
    (55, "C", "Average"),
    (40, "D", "Below Average"),
    (0, "F", "Poor"),
]


def grade_for(score: float) -> tuple[str, str]:
    for threshold, grade, category in GRADE_BANDS:
        if score >= threshold:
            return grade, category
    return "F", "Poor"


def risk_for(score: float) -> str:
    if score < 45:
        return "High"
    if score < 60:
        return "Medium"
    return "Low"


def recommendations(s: dict, final_score: float) -> list[str]:
    tips = []
    if s["attendance"] < 75:
        tips.append(f"Attendance is {s['attendance']:.0f}%. Aim for at least 85%: missed classes are the "
                    "strongest early-warning signal.")
    if s["study_hours"] < 8:
        tips.append(f"Only {s['study_hours']:.1f} study hours/week. Build up to 12-15 hours with short, "
                    "regular sessions rather than cramming.")
    elif s["study_hours"] > 30:
        tips.append("Study load is very high. Returns diminish past ~25 hours/week; protect rest and "
                    "focus on active recall.")
    if s["assignment_score"] < 55:
        tips.append("Assignment scores are low. Review feedback, attend office hours and start "
                    "assignments earlier.")
    if s["participation"] < 4:
        tips.append("Low class participation. Asking one question per class can noticeably improve "
                    "understanding and engagement.")
    if s["sleep_hours"] < 6:
        tips.append("Sleep is under 6 hours. Chronic sleep debt hurts memory and exam performance. "
                    "Target 7-8 hours.")
    elif s["sleep_hours"] > 9.5:
        tips.append("Sleeping more than 9.5 hours regularly may indicate fatigue or low engagement; "
                    "keep a consistent schedule.")
    if s["previous_score"] < 50:
        tips.append("Weak prior results suggest foundation gaps. Consider peer tutoring or a bridging "
                    "course for core topics.")
    if not tips:
        tips.append("Great habits across the board. Keep the current routine and aim for stretch goals."
                    if final_score >= 70 else
                    "No major red flags. Small consistent improvements in study time and assignments "
                    "should lift the score.")
    return tips


class HybridEngine:
    def __init__(self, ann: ANNPredictor, fuzzy: FuzzyPerformanceSystem) -> None:
        self.ann = ann
        self.fuzzy = fuzzy

    def predict_many(self, students: list[dict], detail: bool = False) -> list[dict]:
        """Predict for many students; ANN inference is vectorised."""
        if not students:
            return []
        df = pd.DataFrame([{f: s[f] for f in FEATURES} for s in students])
        ann_scores = self.ann.predict(df)

        results = []
        for s, ann_score in zip(students, ann_scores):
            ann_score = float(ann_score)
            w = ANN_WEIGHT
            fz = self.fuzzy.infer({
                "ann_score": ann_score,
                "attendance": s["attendance"],
                "study_hours": s["study_hours"],
                "assignment_score": s["assignment_score"],
                "participation": s["participation"],
            })
            final = w * ann_score + (1 - w) * fz["score"]
            grade, category = grade_for(final)
            res = {
                "student_name": s.get("student_name"),
                "inputs": {f: s[f] for f in FEATURES},
                "ann_score": round(ann_score, 2),
                "fuzzy_score": fz["score"],
                "fuzzy_label": fz["label"],
                "final_score": round(final, 2),
                "grade": grade,
                "category": category,
                "risk_level": risk_for(final),
                "will_pass": bool(final >= 40),
                "model_agreement": round(max(0.0, 100 - abs(ann_score - fz["score"])), 1),
            }
            if detail:
                res["recommendations"] = recommendations(s, final)
                res["fuzzy_details"] = {
                    k: fz[k] for k in
                    ("input_memberships", "output_memberships", "fired_rules", "aggregated")
                }
            results.append(res)
        return results

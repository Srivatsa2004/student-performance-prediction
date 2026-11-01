"""Mamdani fuzzy inference system, implemented from scratch with NumPy.

Inputs  : ANN predicted score, attendance, study hours, assignment score,
          class participation.
Output  : a 0-100 "performance index" with five linguistic terms.
Method  : min (AND) -> rule weight -> min-clipping -> max aggregation ->
          centroid defuzzification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def trapmf(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership function (a==b or c==d gives a shoulder).
    A triangle is simply trapmf(x, a, b, b, c)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        m = (x > c) & (x < d)
        y[m] = (d - x[m]) / (d - c)
    return y


@dataclass
class FuzzyVariable:
    name: str
    lo: float
    hi: float
    terms: dict  # term name -> (a, b, c, d)

    def fuzzify(self, x: float) -> dict:
        x = float(np.clip(x, self.lo, self.hi))
        return {t: float(trapmf(x, *p)[0]) for t, p in self.terms.items()}

    def curves(self, n: int = 101) -> dict:
        xs = np.linspace(self.lo, self.hi, n)
        return {
            "x": xs.tolist(),
            "terms": {t: trapmf(xs, *p).tolist() for t, p in self.terms.items()},
        }


@dataclass
class FuzzyRule:
    antecedents: dict  # variable -> term (combined with AND)
    consequent: str
    weight: float = 1.0

    @property
    def text(self) -> str:
        cond = " AND ".join(f"{v.replace('_', ' ')} is {t}" for v, t in self.antecedents.items())
        return f"IF {cond} THEN performance is {self.consequent.replace('_', ' ')}"


def _r(then: str, weight: float = 1.0, **ifs) -> FuzzyRule:
    return FuzzyRule(ifs, then, weight)


class FuzzyPerformanceSystem:
    def __init__(self) -> None:
        self.inputs = {
            "ann_score": FuzzyVariable("ann_score", 0, 100, {
                "low": (0, 0, 35, 50), "medium": (40, 55, 55, 70), "high": (60, 75, 100, 100)}),
            "attendance": FuzzyVariable("attendance", 0, 100, {
                "low": (0, 0, 40, 65), "medium": (50, 70, 70, 88), "high": (75, 90, 100, 100)}),
            "study_hours": FuzzyVariable("study_hours", 0, 40, {
                "low": (0, 0, 5, 10), "moderate": (6, 12, 12, 20), "high": (15, 22, 40, 40)}),
            "assignment_score": FuzzyVariable("assignment_score", 0, 100, {
                "poor": (0, 0, 40, 55), "average": (45, 62, 62, 78), "good": (68, 82, 100, 100)}),
            "participation": FuzzyVariable("participation", 0, 10, {
                "low": (0, 0, 2, 4), "medium": (2.5, 5, 5, 7.5), "high": (6, 8, 10, 10)}),
        }
        self.output = FuzzyVariable("performance", 0, 100, {
            "poor": (0, 0, 20, 40),
            "below_average": (25, 40, 40, 55),
            "average": (45, 60, 60, 75),
            "good": (65, 77, 77, 90),
            "excellent": (80, 92, 100, 100),
        })
        self.y = np.linspace(0, 100, 401)
        self._out_mf = {t: trapmf(self.y, *p) for t, p in self.output.terms.items()}
        self.rules = self._build_rules()

    @staticmethod
    def _build_rules() -> list[FuzzyRule]:
        return [
            # Strong performers
            _r("excellent", ann_score="high", attendance="high", assignment_score="good"),
            _r("excellent", ann_score="high", participation="high", study_hours="high"),
            _r("excellent", 0.9, study_hours="high", assignment_score="good", participation="high"),
            _r("good", ann_score="high", attendance="high"),
            _r("good", ann_score="high", attendance="medium"),
            _r("good", ann_score="high", assignment_score="good"),
            _r("average", ann_score="high", attendance="low"),
            _r("average", ann_score="high", study_hours="low"),
            # Mid-range students
            _r("good", ann_score="medium", attendance="high", study_hours="high"),
            _r("good", ann_score="medium", assignment_score="good", participation="high"),
            _r("good", ann_score="medium", study_hours="high", assignment_score="good"),
            _r("average", ann_score="medium", attendance="high", study_hours="moderate"),
            _r("average", ann_score="medium", attendance="medium"),
            _r("average", ann_score="medium", assignment_score="average"),
            _r("below_average", ann_score="medium", study_hours="low"),
            _r("below_average", ann_score="medium", assignment_score="poor"),
            _r("below_average", ann_score="medium", attendance="low"),
            # Struggling students
            _r("poor", ann_score="low", attendance="low"),
            _r("poor", ann_score="low", study_hours="low"),
            _r("poor", ann_score="low", assignment_score="poor"),
            _r("average", 0.8, ann_score="low", attendance="high", study_hours="high"),
            _r("below_average", ann_score="low", attendance="medium"),
            _r("below_average", ann_score="low", assignment_score="good"),
            # Behaviour-only rules (independent of the ANN)
            _r("poor", attendance="low", participation="low"),
            _r("poor", study_hours="low", assignment_score="poor"),
            _r("good", 0.8, attendance="high", assignment_score="good", participation="high"),
            # Low-weight baseline rules so the system is always defined
            _r("good", 0.5, ann_score="high"),
            _r("average", 0.5, ann_score="medium"),
            _r("below_average", 0.5, ann_score="low"),
        ]

    # --------------------------------------------------------------- inference
    def infer(self, values: dict) -> dict:
        degrees = {name: var.fuzzify(values[name]) for name, var in self.inputs.items()}
        agg = np.zeros_like(self.y)
        fired = []
        for rule in self.rules:
            strength = min(degrees[v][t] for v, t in rule.antecedents.items()) * rule.weight
            if strength > 1e-6:
                agg = np.maximum(agg, np.minimum(strength, self._out_mf[rule.consequent]))
                fired.append({"rule": rule.text, "strength": round(float(strength), 4),
                              "consequent": rule.consequent})
        fired.sort(key=lambda r: r["strength"], reverse=True)

        if agg.sum() < 1e-9:  # no rule fired -> fall back to the ANN estimate
            score = float(values["ann_score"])
        else:
            score = float((self.y * agg).sum() / agg.sum())  # centroid

        out_deg = self.output.fuzzify(score)
        label = max(out_deg, key=out_deg.get)
        return {
            "score": round(score, 2),
            "label": label,
            "input_memberships": {k: {t: round(d, 4) for t, d in v.items()} for k, v in degrees.items()},
            "output_memberships": {t: round(d, 4) for t, d in out_deg.items()},
            "fired_rules": fired,
            "aggregated": {"x": self.y[::4].tolist(), "y": agg[::4].round(4).tolist()},
        }

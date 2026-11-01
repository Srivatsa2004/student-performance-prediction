import pytest

from backend.ann_model import ANNPredictor
from backend.data_generator import generate_dataset
from backend.fuzzy_system import FuzzyPerformanceSystem, trapmf
from backend.hybrid import HybridEngine, grade_for


@pytest.fixture(scope="module")
def engine():
    ann = ANNPredictor()
    ann.train(generate_dataset(1500), hidden_layers=(32, 16), max_iter=200)
    return HybridEngine(ann, FuzzyPerformanceSystem())


def test_membership_shapes():
    assert trapmf(0, 0, 0, 10, 20)[0] == 1.0       # left shoulder
    assert trapmf(100, 60, 80, 100, 100)[0] == 1.0  # right shoulder
    assert trapmf(15, 0, 10, 10, 20)[0] == pytest.approx(0.5)


def test_fuzzy_extremes():
    f = FuzzyPerformanceSystem()
    strong = f.infer(dict(ann_score=92, attendance=98, study_hours=25, assignment_score=92, participation=9))
    weak = f.infer(dict(ann_score=20, attendance=30, study_hours=2, assignment_score=25, participation=1))
    assert strong["score"] > 80 and weak["score"] < 30
    assert strong["fired_rules"] and weak["fired_rules"]


def test_ann_quality():
    ann = ANNPredictor()
    meta = ann.train(generate_dataset(2000), hidden_layers=(32, 16), max_iter=300)
    assert meta["metrics"]["r2"] > 0.85


def test_hybrid_orders_students(engine):
    good = dict(attendance=95, study_hours=20, previous_score=88, assignment_score=90, participation=9, sleep_hours=7.5)
    bad = dict(attendance=40, study_hours=2, previous_score=35, assignment_score=35, participation=1, sleep_hours=5)
    g, b = engine.predict_many([good, bad])
    assert g["final_score"] > b["final_score"] + 30
    assert g["grade"] in ("A", "B") and b["grade"] in ("D", "F")


def test_grade_bands():
    assert grade_for(90)[0] == "A" and grade_for(39.9)[0] == "F"

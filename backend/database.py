"""Tiny SQLite store for prediction history."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                student_name TEXT,
                inputs TEXT NOT NULL,
                ann_score REAL, fuzzy_score REAL, final_score REAL,
                grade TEXT, category TEXT, risk_level TEXT
            )"""
        )


def log_prediction(res: dict) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO history (created_at, student_name, inputs, ann_score, fuzzy_score, "
            "final_score, grade, category, risk_level) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), res.get("student_name"),
             json.dumps(res["inputs"]), res["ann_score"], res["fuzzy_score"], res["final_score"],
             res["grade"], res["category"], res["risk_level"]),
        )


def fetch_history(limit: int = 200) -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d.update(json.loads(d.pop("inputs")))
        out.append(d)
    return out


def clear_history() -> int:
    with _conn() as con:
        return con.execute("DELETE FROM history").rowcount

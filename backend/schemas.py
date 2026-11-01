"""Pydantic request models."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class StudentInput(BaseModel):
    student_name: Optional[str] = Field(None, max_length=100)
    attendance: float = Field(..., ge=0, le=100)
    study_hours: float = Field(..., ge=0, le=40)
    previous_score: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    participation: float = Field(..., ge=0, le=10)
    sleep_hours: float = Field(..., ge=0, le=14)


class BatchRequest(BaseModel):
    students: list[StudentInput] = Field(..., min_length=1, max_length=5000)


class TrainRequest(BaseModel):
    source: Literal["synthetic", "custom"] = "synthetic"
    n_samples: int = Field(3000, ge=200, le=50000)
    hidden_layers: list[int] = Field(default_factory=lambda: [64, 32], min_length=1, max_length=5)
    max_iter: int = Field(600, ge=50, le=3000)
    seed: int = 42
    records: Optional[list[dict]] = None

    @field_validator("hidden_layers")
    @classmethod
    def _check_layers(cls, v):
        if any(h < 2 or h > 512 for h in v):
            raise ValueError("each hidden layer must have between 2 and 512 neurons")
        return v

"""Pydantic request/response models for the EnginePulse API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class WindowRequest(BaseModel):
    """One sensor window: list of timesteps, each a list of sensor values.

    Shape must match the loaded model's (window_len, n_features); the server
    checks it and returns a clear 422 rather than a cryptic numpy error.
    """

    model_config = {"json_schema_extra": {
        "example": {"windows": [[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]], "with_interval": True}
    }}

    windows: list[list[list[float]]] = Field(..., min_length=1, max_length=256,
                                             description="Batch of windows, each (timesteps x sensors).")
    with_interval: bool = Field(False, description="Include a 90% prediction interval when supported.")

    @field_validator("windows")
    @classmethod
    def _rectangular(cls, v: list[list[list[float]]]):
        if not v or not v[0]:
            raise ValueError("empty window")
        w, f = len(v[0]), len(v[0][0])
        for win in v:
            if len(win) != w or any(len(t) != f for t in win):
                raise ValueError(f"ragged batch: expected every window to be {w}x{f}")
        return v


class Prediction(BaseModel):
    rul_cycles: float = Field(..., ge=0.0, description="Predicted remaining useful life (cycles).")
    lower_90: float | None = None
    upper_90: float | None = None


class PredictResponse(BaseModel):
    model: str
    dataset: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    key: str          # what you pass to --model / load dir name
    dataset: str
    seed: int
    window_len: int
    n_features: int


class HealthResponse(BaseModel):
    status: str
    version: str
    n_models_loaded: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(4, ge=1, le=20)


class Citation(BaseModel):
    document: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refusal: bool = Field(..., description="True when the corpus does not contain the answer.")

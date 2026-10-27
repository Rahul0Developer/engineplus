"""FastAPI application factory.

Endpoints:
    GET  /health       liveness + how many models are loaded
    GET  /models       list loadable model keys
    POST /predict      raw sensor window -> RUL (+ optional 90% interval)
    POST /ask          document Q&A (only when a DocQA index is configured)

The app is a factory (create_app) rather than a module-level singleton so the
TestClient can inject its own registry, and production can pass an API key.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
except ImportError as exc:  # pragma: no cover
    raise ImportError("the API needs `pip install 'enginepulse[api]'`") from exc

from enginepulse import __version__
from enginepulse.api.dependencies import LoadedModel, ModelRegistry
from enginepulse.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    ModelInfo,
    PredictResponse,
    Prediction,
    WindowRequest,
)
from enginepulse.utils import get_logger

logger = get_logger(__name__)


def create_app(
    models_dir: str = "benchmarks/results/artifacts",
    api_key: Optional[str] = None,
    docqa_engine=None,
) -> "FastAPI":
    """Build the app. `api_key=None` disables auth (fine for local use)."""
    registry = ModelRegistry(models_dir)
    app = FastAPI(
        title="EnginePulse",
        version=__version__,
        description="RUL prediction and maintenance-document Q&A. Research tool — "
                    "not validated for safety-critical decisions.",
    )
    app.state.registry = registry
    app.state.docqa = docqa_engine
    app.state.auth_key = api_key or os.environ.get("ENGINEPULSE_API_KEY") or None

    @app.on_event("startup")
    def _startup() -> None:
        registry.load_all()

    async def verify_key(x_api_key: Optional[str] = Header(None)) -> None:
        if app.state.auth_key and x_api_key != app.state.auth_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__,
                              n_models_loaded=len(registry.keys()))

    @app.get("/models", response_model=list[ModelInfo], dependencies=[Depends(verify_key)])
    def models() -> list[ModelInfo]:
        out = []
        for k in registry.keys():
            lm = registry.get(k)
            out.append(ModelInfo(key=lm.key, dataset=lm.dataset, seed=lm.seed,
                                 window_len=lm.window_len, n_features=lm.n_features))
        return out

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_key)])
    def predict(req: WindowRequest, request: Request, model: Optional[str] = None) -> PredictResponse:
        try:
            lm: LoadedModel = registry.get(model)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        X = np.asarray(req.windows, dtype=np.float32)
        if X.shape[1] != lm.window_len or X.shape[2] != lm.n_features:
            raise HTTPException(
                status_code=422,
                detail=(f"window shape {X.shape[1:]} does not match model context "
                        f"(expected {lm.window_len} timesteps x {lm.n_features} sensors). "
                        "Check which model/dataset you trained on."),
            )
        Xn = lm.normalise(X)
        try:
            preds = lm.model.predict(Xn)
        except ValueError as exc:
            # Feature-width mismatches etc. — a client-shape problem, not a crash.
            raise HTTPException(status_code=422, detail=f"model rejected input: {exc}") from exc

        lo = hi = None
        if req.with_interval:
            try:
                q = lm.model.predict_quantiles(Xn, [0.05, 0.95])
                lo, hi = q[0.05], q[0.95]
                if np.isnan(lo).any():
                    lo = hi = None
            except (RuntimeError, NotImplementedError):
                lo = hi = None
            if lo is None:
                # No native quantiles: fall back to frozen validation-residual
                # bounds shipped alongside the artifact (see trainer).
                b = getattr(lm, "residual_bounds", None)
                if b is not None:
                    lo = np.clip(preds + b[0], 0.0, lm.max_rul or np.inf)
                    hi = np.clip(preds + b[1], 0.0, lm.max_rul or np.inf)

        items = []
        for i, p in enumerate(preds):
            items.append(Prediction(
                rul_cycles=float(p),
                lower_90=float(lo[i]) if lo is not None else None,
                upper_90=float(hi[i]) if hi is not None else None,
            ))
        return PredictResponse(model=lm.key, dataset=lm.dataset, predictions=items)

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_key)])
    def ask(req: AskRequest) -> AskResponse:
        engine = app.state.docqa
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="DocQA not configured. Start with --docqa-dir or see 'enginepulse ask'.",
            )
        res = engine.ask(req.question, k=req.k)
        return AskResponse(
            answer=res["answer"],
            refusal=res["refusal"],
            citations=[Citation(**c) for c in res["citations"]],
        )

    return app


def main() -> None:  # pragma: no cover - convenience runner
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Run the EnginePulse API server")
    ap.add_argument("--models-dir", default="benchmarks/results/artifacts")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(args.models_dir), host=args.host, port=args.port)

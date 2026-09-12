"""Thin HTTP wrapper around QuadricSolver.

The browser owns rendering, flying, and silhouette extraction. This process
owns only the geometric solver so the ARM port can lift solver/ verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

from typing import Any

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver import QuadricSolver  # noqa: E402

WEB = ROOT / "web"

app = FastAPI(title="QuadricSLAM lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SOLVER = QuadricSolver()


@app.post("/api/reset")
def api_reset() -> dict:
    SOLVER.reset()
    return {"ok": True, **SOLVER.estimate_dict()}


@app.post("/api/observe")
def api_observe(payload: dict[str, Any] = Body(...)) -> dict:
    SOLVER.observe_payload(payload)
    return {"ok": True, **SOLVER.estimate_dict()}


@app.get("/api/estimate")
def api_estimate() -> dict:
    return {"ok": True, **SOLVER.estimate_dict()}


@app.get("/api/health")
def api_health() -> dict:
    return {"ok": True, "initialized": SOLVER.initialized, "n": len(SOLVER.observations)}


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")

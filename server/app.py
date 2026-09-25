"""Static files and a measurement-stream relay.

The browser publishes quadricslam.sim.v1 frames. Any number of consumers
subscribe on the same socket. solver/ is not imported here.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SCHEMA = "quadricslam.sim.v1"
GONE = {"ok": False, "error": "use /ws/stream"}

app = FastAPI(title="QuadricSLAM lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_clients: set[WebSocket] = set()
_frames = 0
_recent: deque[str] = deque(maxlen=512)


def _accept_frame(text: str) -> bool:
    global _frames
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(msg, dict) or msg.get("schema") != SCHEMA:
        return False
    _frames += 1
    _recent.append(text)
    return True


async def _fanout(text: str, skip: WebSocket | None = None) -> None:
    dead: list[WebSocket] = []
    for ws in list(_clients):
        if ws is skip:
            continue
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@app.get("/api/health")
def api_health() -> dict:
    return {"ok": True, "clients": len(_clients), "frames": _frames}


@app.post("/api/frame", response_model=None)
async def api_frame(payload: dict[str, Any] = Body(...)) -> dict | JSONResponse:
    text = json.dumps(payload, separators=(",", ":"))
    if not _accept_frame(text):
        return JSONResponse(
            {"ok": False, "error": "expected quadricslam.sim.v1"},
            status_code=400,
        )
    await _fanout(text)
    return {"ok": True, "frames": _frames}


@app.get("/api/recording.ndjson")
def api_recording() -> Response:
    body = "\n".join(_recent)
    if body:
        body += "\n"
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/observe")
def api_observe() -> JSONResponse:
    return JSONResponse(GONE, status_code=410)


@app.post("/api/reset")
def api_reset() -> JSONResponse:
    return JSONResponse(GONE, status_code=410)


@app.get("/api/estimate")
def api_estimate() -> JSONResponse:
    return JSONResponse(GONE, status_code=410)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            text = await ws.receive_text()
            if _accept_frame(text):
                await _fanout(text, skip=ws)
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")

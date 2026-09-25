# QuadricSLAM lab

Sensor and scene simulator for QuadricSLAM-style measurements. The browser
flies a camera around a known object, extracts 2D silhouettes, and streams
ground truth plus measurements. `solver/` is an offline geometry library. The
page does not run it, and the server does not call it.

## Run

```bash
pip install -r requirements.txt
./run.sh          # http://127.0.0.1:8765
```

```bash
python3 tests/test_solver.py
node --test tests/test_trajectory.mjs tests/test_mask.mjs
```

Live stream (browser publishes, any number of subscribers):

- `ws://127.0.0.1:8765/ws/stream`
- `GET /api/health` → `{ "ok", "clients", "frames" }`
- `POST /api/frame` injects one `quadricslam.sim.v1` object
- `GET /api/recording.ndjson` is the server's last 512 frames
- the panel can also record and download NDJSON locally

`POST /api/observe` returns 410. A subscriber prints `seq`, `kind`, and the
camera translation with:

```python
import asyncio, json, websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:8765/ws/stream") as ws:
        await ws.send(json.dumps({"type": "hello", "role": "subscriber"}))
        async for raw in ws:
            m = json.loads(raw)
            if m.get("schema") != "quadricslam.sim.v1":
                continue
            T = m["gt"]["T_world_from_cam"]
            print(m["seq"], m["kind"], [round(T[i][3], 4) for i in range(3)])
asyncio.run(main())
```

## GitHub Pages

The browser UI in `web/` is static (Three.js from jsDelivr). GitHub Pages can
host the fly-and-extract view. The WebSocket relay cannot run on Pages — live
frames need `./run.sh`.

This repo deploys `web/` with `.github/workflows/pages.yml`. One-time setup:

1. Repo Settings → Pages → Source: **GitHub Actions**
2. Push to `main` (or run the workflow manually)

Site URL: `https://matthewgiarra.github.io/quadricslam-lab/`

Asset paths are relative (`style.css`, `app.js`), so the same files work under
FastAPI at `/` and under a project-pages subpath. Optional `?api=https://host`
points the page at a remote relay. A Pages origin is HTTPS, so
`ws://127.0.0.1` is blocked by mixed content.

## What you can vary

- Measurement type: axis-aligned box, oriented box, convex hull, or segmentation mask
- Detection rate 0.1–30 Hz, independent of the display framerate
- Focal length, stereo baseline, and sensor format. Focal length, baseline, and the orbit numbers can be typed, and each slider’s ends are editable
- Pixel jitter (including a translation of the segmentation mask), disparity noise, box dropout, independent depth dropout
- Manual WASD, or a canned orbit (plane, semi-major axis, eccentricity, object-minus-center offset in meters, period, ingress, look-at-object or tangent)
- Ground-truth ellipsoid, axes, and camera path (green). Yellow is the last detection.
- Built-in ellipsoid / box, or load a GLB (Draco / meshopt / KTX2). A loaded model’s longest side is a length in meters
- The view stays lit out to 5000 km. Reset view returns the camera to the start

## Frame

Each detection is one JSON object, `schema` = `quadricslam.sim.v1`. Poses are
row-major 4×4. `gt.T_world_from_cam` is the left camera in the OpenCV frame
(X right, Y down, Z forward), converted from Three.js with `D = diag(1,-1,-1)`.
`meas.left` / `meas.right` carry only the active `kind` (`aabb`, `obb`, `hull`,
or `mask`). A mask is the filled silhouette, 256×256, `encoding`
`row-major-bits-lsb`, `scale_to_image` 4. Jitter translates that silhouette;
it is not a convex hull.

## Solver notes

`solver/` is unchanged and still importable. Run it from `tests/test_solver.py`
or your own consumer — not from this process.

- The simulator world is inertial. The solver's own gauge, when you use it
  offline, still locks the first left camera.
- Scale comes from the stereo baseline; depth residual is box-center disparity.
- Sliding active set (first views + tail + uniform subsample) so a long flight
  does not grow the LM without bound.
- Each observation stores its own `K`, so you can move the focal-length slider
  mid-run without rewriting history.

## Porting to the LS1046

Take `solver/geometry.py` and `solver/optimizer.py`. The state vector is
`[rotvec_ell(3), t_ell(3), log_radii(3)] + [rotvec_cam(3), t_cam(3)] * (N-1)`.
Residuals are AABB (4), OBB (5), hull-line (`l^T C* l`), and disparity (1).
Replace `scipy.optimize.least_squares` with a small Eigen Levenberg–Marquardt.

A detector publishes `quadricslam.sim.v1` on `/ws/stream`. `POST /api/observe`
is gone (410).

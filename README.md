# QuadricSLAM lab

Interactive testbed for the **geometry downstream of a detector** in QuadricSLAM-style
object SLAM: joint estimation of a constrained dual quadric (ellipsoid) and a
stereo-rig trajectory from 2D measurements.

The Python package `solver/` is deliberately free of UI code so it can move to
Eigen/C++ on the Teledyne LS1046-Space later. The browser only flies a camera
and extracts silhouettes.

## Run

```bash
pip install -r requirements.txt
./run.sh          # http://127.0.0.1:8765
```

```bash
python3 tests/test_solver.py
```

## GitHub Pages

The browser UI in `web/` is static (Three.js from jsDelivr). GitHub Pages can
host it. The Python solver cannot run on Pages — that origin is fly / load-GLB
/ extract-silhouette only. Live estimates still need `./run.sh`.

This repo deploys `web/` with `.github/workflows/pages.yml`. One-time setup:

1. Repo Settings → Pages → Source: **GitHub Actions**
2. Push to `main` (or run the workflow manually)

Site URL: `https://matthewgiarra.github.io/quadricslam-lab/`

Asset paths are relative (`style.css`, `app.js`), so the same files work under
FastAPI at `/` and under a project-pages subpath. Optional `?api=https://host`
points the UI at a remote solver; a Pages page is HTTPS, so `http://127.0.0.1`
is blocked by mixed content.

## What you can vary

- Measurement type: axis-aligned box, oriented box, convex hull of the mask
- Detection rate 0.1–30 Hz, independent of the display framerate
- Focal length, stereo baseline, sensor format
- Pixel jitter, disparity noise, box dropout, independent depth dropout
- GT vs estimated ellipsoid, axes, and camera path (colors: green GT, cyan estimate)
- Reset estimator without moving the camera
- Built-in ellipsoid / box, or load a GLB (Draco / meshopt compressed files are supported)

## Solver notes

- World frame is locked to the first left camera (gauge).
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

A real detector later implements the same JSON the browser already posts to
`POST /api/observe`.

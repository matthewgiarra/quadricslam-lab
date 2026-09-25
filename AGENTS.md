# QuadricSLAM lab — agent rules

This repo is a **sensor / scene simulator**, not an estimator host.

The browser flies a camera around a known object, extracts 2D measurements
(AABB / OBB / convex hull, optional silhouette points), and **streams ground
truth plus measurements** to an external consumer. QuadricSLAM, visualization
of estimates, and comparison to GT live in a **separate process**. Do not put
the estimator back into `web/` or couple the UI to `/api/observe`.

## Target split

| Piece | Owns |
|---|---|
| `web/` | Three.js scene, GLB load, silhouette extraction, canned trajectories, HUD, publish frames |
| `server/` | Static files + thin stream relay (WebSocket / SSE). No solver. |
| `solver/` | Existing geometry / LM code. Keep importable. Do not call it from the browser. |
| `consumer/` (new) | Connects to the stream, runs estimators, plots estimate vs GT |

If `consumer/` does not exist yet, create it only when asked. First work is
the simulator + stream + canned trajectories.

## Frames and conventions (do not invent new ones)

- Display / Three.js camera is Y-up, look-toward −Z.
- OpenCV / solver frame is X-right, Y-down, Z-forward.
- Existing conversion: `D_CV = diag(1, −1, −1)` applied to `camera.matrixWorld`.
  Stream **OpenCV** poses as the primary GT (`T_world_from_cam`, row-major 4×4).
  You may also include Three.js position/quaternion for the UI, labeled as such.
- Sensor image is square **1024×1024**. Detector render is **256×256**, boxes
  scaled back to 1024. Viewport is letterboxed square to match the sensor.
- Intrinsics: `fx = fy = (focal_mm / sensor_mm) * 1024`, `cx = cy = 512`.
- Stereo is rectified horizontal; baseline along +X in the OpenCV camera frame.
- Object default pose is at the world origin. World is an inertial scene frame,
  not “first-camera gauge” (that gauge is an estimator choice).
- Chaser **attitude is treated as known** (star trackers). The stream still
  includes full GT pose so a consumer can lock R and estimate translation +
  quadric, or ignore that assumption.

## Stream contract

Every detection tick emits one JSON object (see skill `frame-protocol`).
Required fields:

- `schema`, `seq`, `t` (seconds, monotonic)
- `K`, `baseline_m`, `image_size`
- `kind`: `aabb` | `obb` | `hull`
- `meas.left` / `meas.right` matching `kind`
- `gt.T_world_from_cam` (left camera, OpenCV, 4×4 row-major)
- `gt.T_world_from_object` (4×4 row-major)
- `gt.ellipsoid` if the primitive is an ellipsoid (`radii`, `R`, `t`)
- `traj` snapshot (mode + parameters actually in use)

Transport: WebSocket from the FastAPI process, default
`ws://127.0.0.1:8765/ws/stream`. Browser publishes; any number of consumers
subscribe. Also support downloading an NDJSON recording.

Do **not** POST measurements to `/api/observe`. That endpoint is legacy.
Replace it with the stream (keep a stub only if something still calls it).

## Canned trajectories

Implement parametric paths in the **world frame**, object-relative:

- Mode: `manual` (current WASD fly) or `canned`
- Path family: `orbit` (ellipse in a plane) with a **smooth ingress**
  (C1 position, no snap) from the current camera pose onto the orbit
- Parameters the panel must expose:
  - plane orientation (two angles or a unit normal)
  - semi-major axis, eccentricity in `[0, 0.95)`
  - object offset from the ellipse center (3-vector, world)
  - orbit period / rate
  - look mode: `at_object` (default) or `tangent`
  - ingress duration
- Camera up should stay close to the orbit-plane normal (or world +Y if the
  plane is near-horizontal) so the view does not corkscrew unless asked.

Keep trajectory math in its own module (`web/trajectory.js`), not inlined
inside the render loop.

## UI rules

- Yellow overlay = last detection. Do not block the overlay on a solver round-trip.
- Detection timer is independent of the display loop (`DET=256`).
- Letterbox the view to a square matching the 1024 sensor.
- GLB load: `GLTFLoader` + `DRACOLoader` (three@0.160.0 jsDelivr `draco/gltf`)
  + `MeshoptDecoder` + KTX2. Do not drop compression support.
- Open the lab at **http://127.0.0.1:8765** even if uvicorn binds `0.0.0.0`.
- Asset paths stay relative so GitHub Pages still works. Live stream needs `./run.sh`.

## How to run

```bash
pip install -r requirements.txt
./run.sh                 # http://127.0.0.1:8765
python3 tests/test_solver.py
```

`run.sh` is `uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8765} --reload`.

## Coding conventions

- Browser: vanilla ES modules, Three r160 via import map. No React/bundler
  unless explicitly requested.
- Python: 3.11+, type hints, no UI imports inside `solver/`.
- Prefer small diffs that compile. Do not rewrite `solver/` while changing
  the browser into a stream source.
- Do not force-push. Local git history and GitHub main differ; never
  `git push --force` to `origin`.
- After behavior changes, update `README.md` only if the run/API story changed.

## What not to do

- Do not re-introduce estimated-ellipsoid / estimated-path overlays that
  depend on a live solver in this process.
- Do not encode scale from monocular VO. Scale comes from stereo/range.
- Do not change the 1024 image convention or the OpenCV/Three conversion
  without updating the stream schema version.
- Do not add README / env / scaffolding files unless needed for the task.

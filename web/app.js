import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";

// glTF-Transform / many exporters mark these as required:
// KHR_draco_mesh_compression and KHR_texture_basisu (KTX2).
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0";
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(`${THREE_CDN}/examples/jsm/libs/draco/gltf/`);
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);
gltfLoader.setMeshoptDecoder(MeshoptDecoder);

const IMAGE = 1024;
// Detector render is much smaller than the display; boxes are scaled back to IMAGE.
const DET = 256;
const DET_SCALE = IMAGE / DET;
const D_CV = new THREE.Matrix4().makeScale(1, -1, -1);

// Relative API so the same files work under FastAPI (`/`) and GitHub Pages
// (`/quadricslam-lab/`). Optional `?api=https://host` points at a remote solver.
// A Pages origin is HTTPS, so a plain http://127.0.0.1 solver is blocked by
// mixed content — run `./run.sh` for the live estimator.
const API_BASE = (() => {
  const q = new URLSearchParams(location.search).get("api");
  if (q) return q.replace(/\/$/, "");
  return "";
})();
const apiUrl = (path) => `${API_BASE}${path}`;
let solverReachable = false;

const $ = (id) => document.getElementById(id);
const state = {
  kind: "aabb",
  hz: 5,
  focalMm: 50,
  sensorMm: 11.313708,
  baseline: 0.10,
  jitter: 0,
  dnoise: 0,
  drop: 0,
  ddrop: 0,
  armed: true,
  lastDetect: 0,
  pending: false,
  gtPath: [],
  estPath: [],
  lastDet: null,
  estimate: null,
};

const viewEl = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(viewEl.clientWidth, viewEl.clientHeight);
renderer.setClearColor(0x0b0d12, 1);
viewEl.appendChild(renderer.domElement);

const ktx2Loader = new KTX2Loader();
ktx2Loader.setTranscoderPath(`${THREE_CDN}/examples/jsm/libs/basis/`);
ktx2Loader.detectSupport(renderer);
gltfLoader.setKTX2Loader(ktx2Loader);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0b0d12, 8, 28);

const camera = new THREE.PerspectiveCamera(30, 1, 0.02, 80);
camera.position.set(2.6, 0.55, 0.4);
camera.lookAt(0, 0, 0);

const miniEl = $("minimap");
const miniRenderer = new THREE.WebGLRenderer({ antialias: true });
miniRenderer.setPixelRatio(1);
miniRenderer.setClearColor(0x0a0c10, 1);
miniEl.appendChild(miniRenderer.domElement);
const miniScene = new THREE.Scene();
const miniCam = new THREE.PerspectiveCamera(40, 280 / 200, 0.1, 50);
miniCam.position.set(4.2, 3.4, 4.2);
miniCam.lookAt(0, 0, 0);

function applyIntrinsics() {
  // Square sensor: same vertical and horizontal FOV as the 1024×1024 detector.
  const fov = THREE.MathUtils.radToDeg(2 * Math.atan((state.sensorMm / 2) / state.focalMm));
  camera.fov = fov;
  camera.aspect = 1;
  camera.updateProjectionMatrix();
  $("focalV").textContent = `${state.focalMm} mm`;
}
function K() {
  const fx = (state.focalMm / state.sensorMm) * IMAGE;
  return { fx, fy: fx, cx: IMAGE / 2, cy: IMAGE / 2, width: IMAGE, height: IMAGE };
}

scene.add(new THREE.AmbientLight(0x9aa8c7, 0.55));
const key = new THREE.DirectionalLight(0xfff4e5, 1.1);
key.position.set(3, 5, 2);
scene.add(key);
scene.add(new THREE.HemisphereLight(0x6b7a99, 0x1a120c, 0.4));

const grid = new THREE.GridHelper(10, 20, 0x243044, 0x1a2230);
scene.add(grid);
miniScene.add(new THREE.GridHelper(8, 16, 0x243044, 0x1a2230));
miniScene.add(new THREE.AxesHelper(0.4));

const objectRoot = new THREE.Group();
scene.add(objectRoot);

function makeMat(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color, roughness: 0.45, metalness: 0.08, transparent: opacity < 1, opacity,
  });
}

function buildEllipsoidMesh(rx = 0.35, ry = 0.20, rz = 0.15) {
  const g = new THREE.SphereGeometry(1, 48, 32);
  g.scale(rx, ry, rz);
  const m = new THREE.Mesh(g, makeMat(0xc4b5fd));
  m.userData.gtRadii = [rx, ry, rz];
  return m;
}
function buildBoxMesh(sx = 0.50, sy = 0.28, sz = 0.22) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), makeMat(0xfbbf24));
  m.userData.gtRadii = [sx / 2, sy / 2, sz / 2];
  return m;
}

let subject = buildEllipsoidMesh();
objectRoot.add(subject);
let loadedGlb = null;

function gtRadii() {
  if (subject.userData.gtRadii) return subject.userData.gtRadii;
  const box = new THREE.Box3().setFromObject(subject);
  const s = new THREE.Vector3();
  box.getSize(s);
  return [s.x / 2, s.y / 2, s.z / 2];
}

function replaceSubject(mesh) {
  objectRoot.clear();
  subject = mesh;
  objectRoot.add(subject);
  rebuildGtOverlay();
  detDirty = true;
}

$("primitive").addEventListener("change", (e) => {
  const v = e.target.value;
  if (v === "ellipsoid") replaceSubject(buildEllipsoidMesh());
  else if (v === "box") replaceSubject(buildBoxMesh());
  else if (v === "glb" && loadedGlb) replaceSubject(loadedGlb);
});
$("glb").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  $("status").textContent = "loading GLB…";
  $("status").className = "";
  try {
    const gltf = await gltfLoader.loadAsync(url);
    const root = gltf.scene;
    const box = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    root.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    root.scale.multiplyScalar(0.8 / maxDim);
    root.updateMatrixWorld(true);
    const box2 = new THREE.Box3().setFromObject(root);
    const s2 = new THREE.Vector3();
    box2.getSize(s2);
    root.userData.gtRadii = [s2.x / 2, s2.y / 2, s2.z / 2];
    loadedGlb = root;
    $("primitive").value = "glb";
    replaceSubject(root);
    $("status").textContent = `loaded ${file.name}`;
    $("status").className = "ok";
  } catch (err) {
    console.error(err);
    $("status").textContent = "GLB load failed";
    $("status").className = "bad";
  } finally {
    URL.revokeObjectURL(url);
  }
});

// ----- overlays (main view) -----
function wireEllipsoid(radii, color, dashed = false) {
  const group = new THREE.Group();
  const mats = [0, 1, 2].map(() => new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  const rings = [[0, 1], [0, 2], [1, 2]];
  for (let r = 0; r < 3; r++) {
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const t = (i / 64) * Math.PI * 2;
      const p = new THREE.Vector3();
      p.setComponent(rings[r][0], radii[rings[r][0]] * Math.cos(t));
      p.setComponent(rings[r][1], radii[rings[r][1]] * Math.sin(t));
      pts.push(p);
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mats[r]));
  }
  return group;
}
function axisHelper(radii, sat = 1) {
  const g = new THREE.Group();
  const cols = [0xff4d4d, 0x4dff6a, 0x4da6ff];
  for (let i = 0; i < 3; i++) {
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3().setComponent(i, radii[i] * 1.25);
    const m = new THREE.LineBasicMaterial({ color: cols[i] });
    m.opacity = sat;
    m.transparent = sat < 1;
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([p0, p1]), m));
  }
  return g;
}

const gtEllGroup = new THREE.Group();
const estEllGroup = new THREE.Group();
scene.add(gtEllGroup);
scene.add(estEllGroup);

function rebuildGtOverlay() {
  gtEllGroup.clear();
  const r = gtRadii();
  gtEllGroup.add(wireEllipsoid(r, 0x4ade80));
  gtEllGroup.add(axisHelper(r, 1));
}
rebuildGtOverlay();

// detection overlay (2d)
const overlay = document.createElement("canvas");
overlay.id = "det-overlay";
document.body.appendChild(overlay);
const octx = overlay.getContext("2d");

function sensorViewSize() {
  return Math.max(64, Math.min(window.innerWidth, window.innerHeight));
}

function placeOverlayOnViewfinder() {
  const side = sensorViewSize();
  const left = Math.floor((window.innerWidth - side) / 2);
  const top = Math.floor((window.innerHeight - side) / 2);
  overlay.style.left = `${left}px`;
  overlay.style.top = `${top}px`;
  overlay.style.width = `${side}px`;
  overlay.style.height = `${side}px`;
  overlay.width = side;
  overlay.height = side;
}

// ----- FPS controls -----
const vel = new THREE.Vector3();
const keys = new Set();
let locked = false;
const euler = new THREE.Euler(0, 0, 0, "YXZ");
euler.setFromQuaternion(camera.quaternion);

viewEl.addEventListener("click", () => renderer.domElement.requestPointerLock());
document.addEventListener("pointerlockchange", () => {
  locked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener("mousemove", (e) => {
  if (!locked) return;
  euler.y -= e.movementX * 0.0022;
  euler.x -= e.movementY * 0.0022;
  euler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, euler.x));
  camera.quaternion.setFromEuler(euler);
});
document.addEventListener("keydown", (e) => {
  keys.add(e.code);
  if (e.code === "Escape") document.exitPointerLock();
});
document.addEventListener("keyup", (e) => keys.delete(e.code));

function fly(dt) {
  const speed = (keys.has("ShiftLeft") || keys.has("ShiftRight") ? 2.6 : 1.1);
  const f = new THREE.Vector3();
  camera.getWorldDirection(f);
  const r = new THREE.Vector3().crossVectors(f, camera.up).normalize();
  const u = camera.up.clone().normalize();
  const a = new THREE.Vector3();
  if (keys.has("KeyW")) a.add(f);
  if (keys.has("KeyS")) a.sub(f);
  if (keys.has("KeyD")) a.add(r);
  if (keys.has("KeyA")) a.sub(r);
  if (keys.has("KeyE")) a.add(u);
  if (keys.has("KeyQ")) a.sub(u);
  if (a.lengthSq() > 0) a.normalize().multiplyScalar(speed);
  vel.lerp(a, 1 - Math.exp(-8 * dt));
  camera.position.addScaledVector(vel, dt);
}

// ----- silhouette extraction -----
const detTarget = new THREE.WebGLRenderTarget(DET, DET);
const detScene = new THREE.Scene();
const detCam = camera.clone();
const whiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
let detRoot = null;
let detDirty = true;

function syncDetScene() {
  if (!detDirty && detRoot) return;
  detScene.clear();
  detRoot = subject.clone(true);
  detRoot.traverse((o) => {
    if (o.isMesh) o.material = whiteMat;
  });
  detScene.add(detRoot);
  detDirty = false;
}

const pixelBuf = new Uint8Array(DET * DET * 4);

function collectMask(needPts) {
  renderer.readRenderTargetPixels(detTarget, 0, 0, DET, DET, pixelBuf);
  const pts = [];
  let xmin = DET, ymin = DET, xmax = 0, ymax = 0;
  let hit = 0;
  // WebGL origin is bottom-left; flip Y into image / OpenCV space.
  for (let y = 0; y < DET; y++) {
    for (let x = 0; x < DET; x++) {
      const srcY = DET - 1 - y;
      const i = (srcY * DET + x) * 4;
      if (pixelBuf[i] > 20) {
        hit++;
        if (x < xmin) xmin = x;
        if (y < ymin) ymin = y;
        if (x > xmax) xmax = x;
        if (y > ymax) ymax = y;
        if (needPts) pts.push(x * DET_SCALE, y * DET_SCALE);
      }
    }
  }
  if (hit < 4) return null;
  return {
    pts,
    xmin: xmin * DET_SCALE,
    ymin: ymin * DET_SCALE,
    xmax: xmax * DET_SCALE,
    ymax: ymax * DET_SCALE,
  };
}

function convexHull(pointsXY) {
  const n = pointsXY.length / 2;
  const pts = [];
  const step = Math.max(1, Math.floor(n / 400));
  for (let i = 0; i < n; i += step) pts.push([pointsXY[2 * i], pointsXY[2 * i + 1]]);
  pts.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function obbFromHull(hull) {
  let best = { area: Infinity };
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(-ang), s = Math.sin(-ang);
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (const p of hull) {
      const x = p[0] * c - p[1] * s;
      const y = p[0] * s + p[1] * c;
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    const area = (maxx - minx) * (maxy - miny);
    if (area < best.area) best = { area, ang, minx, maxx, miny, maxy };
  }
  const cxR = (best.minx + best.maxx) / 2;
  const cyR = (best.miny + best.maxy) / 2;
  const c = Math.cos(best.ang), s = Math.sin(best.ang);
  return {
    cx: cxR * c + cyR * s,
    cy: -cxR * s + cyR * c,
    width: best.maxx - best.minx,
    height: best.maxy - best.miny,
    theta: best.ang,
  };
}

function jitterBox(aabb, rng) {
  const j = state.jitter;
  if (j <= 0) return aabb;
  const n = () => (rng() * 2 - 1) * j;
  return {
    xmin: aabb.xmin + n(), ymin: aabb.ymin + n(),
    xmax: aabb.xmax + n(), ymax: aabb.ymax + n(),
  };
}

function featuresFromMask(mask, rng, kind) {
  const aabb0 = { xmin: mask.xmin, ymin: mask.ymin, xmax: mask.xmax, ymax: mask.ymax };
  const aabb = jitterBox(aabb0, rng);
  if (kind === "aabb") {
    return { aabb, obb: null, hull: { vertices: [] } };
  }
  const hull = convexHull(mask.pts);
  let jhull = hull;
  if (state.jitter > 0) {
    jhull = hull.map((p) => [
      p[0] + (rng() * 2 - 1) * state.jitter,
      p[1] + (rng() * 2 - 1) * state.jitter,
    ]);
  }
  const obb = obbFromHull(jhull.length >= 3 ? jhull : hull);
  return { aabb, obb, hull: { vertices: jhull } };
}

function mulRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function renderSilhouetteAt(cam) {
  detCam.copy(cam, true);
  detCam.aspect = 1;
  const fov = THREE.MathUtils.radToDeg(2 * Math.atan((state.sensorMm / 2) / state.focalMm));
  detCam.fov = fov;
  detCam.updateProjectionMatrix();
  renderer.setRenderTarget(detTarget);
  renderer.setClearColor(0x000000, 1);
  renderer.render(detScene, detCam);
  renderer.setRenderTarget(null);
  renderer.setClearColor(0x0b0d12, 1);
  const needPts = state.kind !== "aabb";
  return collectMask(needPts);
}

function tickDetect() {
  if (!state.armed) return;
  if (Math.random() < state.drop) return;
  syncDetScene();
  const rng = mulRng((performance.now() * 1000) | 0);

  const maskL = renderSilhouetteAt(camera);
  if (!maskL) return;

  const saved = camera.position.clone();
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  camera.position.addScaledVector(right, state.baseline);
  camera.updateMatrixWorld();
  const maskR = renderSilhouetteAt(camera);
  camera.position.copy(saved);
  camera.updateMatrixWorld();
  if (!maskR) return;

  const fL = featuresFromMask(maskL, rng, state.kind);
  const fR = featuresFromMask(maskR, rng, state.kind);
  if (state.dnoise > 0) {
    const dd = (rng() * 2 - 1) * state.dnoise;
    fR.aabb.xmin -= dd; fR.aabb.xmax -= dd;
    if (fR.obb) fR.obb.cx -= dd;
    fR.hull.vertices = fR.hull.vertices.map((p) => [p[0] - dd, p[1]]);
  }
  const kind = state.kind;
  const left = kind === "aabb" ? { aabb: fL.aabb } : kind === "obb" ? { obb: fL.obb } : { hull: fL.hull };
  const rightMeas = kind === "aabb" ? { aabb: fR.aabb } : kind === "obb" ? { obb: fR.obb } : { hull: fR.hull };
  state.lastDet = { kind, left: fL, right: fR };

  if (!solverReachable || state.pending) return;

  const payload = {
    kind,
    stamp: performance.now() / 1000,
    K: K(),
    baseline: state.baseline,
    left,
    right: rightMeas,
    pixel_sigma: Math.max(state.jitter, 1.5),
    disparity_sigma: Math.max(state.dnoise, 1.5),
    has_depth: Math.random() >= state.ddrop,
  };
  state.pending = true;
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), 2500);
  fetch(apiUrl("/api/observe"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: ac.signal,
  })
    .then((res) => res.json())
    .then((est) => {
      state.estimate = est;
      updateEstimateVisuals(est);
      $("status").textContent = est.initialized ? "solver live" : "waiting for stereo lock";
      $("status").className = est.initialized ? "ok" : "bad";
      $("nobs").textContent = est.n_observations ?? 0;
      $("rms").textContent = est.rms != null ? est.rms.toFixed(2) : "—";
    })
    .catch(() => {
      $("status").textContent = solverReachable ? "solver slow/unreachable" : "fly-only (no solver)";
      $("status").className = "bad";
    })
    .finally(() => {
      clearTimeout(t);
      state.pending = false;
    });
}

function mat4FromRowMajor(a) {
  const m = new THREE.Matrix4();
  m.set(
    a[0][0], a[0][1], a[0][2], a[0][3],
    a[1][0], a[1][1], a[1][2], a[1][3],
    a[2][0], a[2][1], a[2][2], a[2][3],
    a[3][0], a[3][1], a[3][2], a[3][3],
  );
  return m;
}

const estPathLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x22d3ee }),
);
const gtPathLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x4ade80 }),
);
miniScene.add(estPathLine);
miniScene.add(gtPathLine);
const miniSubject = new THREE.Mesh(
  new THREE.SphereGeometry(0.08, 12, 8),
  new THREE.MeshBasicMaterial({ color: 0xc4b5fd }),
);
miniScene.add(miniSubject);

function updateEstimateVisuals(est) {
  estEllGroup.clear();
  if (!est || !est.initialized) return;
  const r = est.ellipsoid.radii;
  const wires = wireEllipsoid(r, 0x22d3ee);
  const ax = axisHelper(r, 0.85);
  estEllGroup.add(wires);
  estEllGroup.add(ax);
  camera.updateMatrixWorld();
  const TcamEll = mat4FromRowMajor(est.T_cam_from_ell);
  const M = new THREE.Matrix4().multiplyMatrices(camera.matrixWorld, D_CV);
  M.multiply(TcamEll);
  estEllGroup.matrixAutoUpdate = false;
  estEllGroup.matrix.copy(M);

  if (est.T_wc_current && est.path_centers_world) {
    const Twc = mat4FromRowMajor(est.T_wc_current);
    const Tbridge = new THREE.Matrix4()
      .multiplyMatrices(camera.matrixWorld, D_CV)
      .multiply(Twc.clone().invert());
    const pts = [];
    for (const c of est.path_centers_world) {
      const v = new THREE.Vector3(c[0], c[1], c[2]).applyMatrix4(Tbridge);
      pts.push(v);
    }
    state.estPath = pts;
    if (pts.length >= 2) {
      estPathLine.geometry.dispose();
      estPathLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    }
  }
}

function pushGtPath() {
  state.gtPath.push(camera.position.clone());
  if (state.gtPath.length > 600) state.gtPath.shift();
  if (state.gtPath.length >= 2) {
    gtPathLine.geometry.dispose();
    gtPathLine.geometry = new THREE.BufferGeometry().setFromPoints(state.gtPath);
  }
}

function drawDetectionOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!$("showDet").checked || !state.lastDet) return;
  const sx = overlay.width / IMAGE;
  const sy = overlay.height / IMAGE;
  const det = state.lastDet.left;
  octx.strokeStyle = "rgba(251,191,36,0.9)";
  octx.lineWidth = 1.5;
  if (state.kind === "aabb" && det.aabb) {
    const a = det.aabb;
    octx.strokeRect(a.xmin * sx, a.ymin * sy, (a.xmax - a.xmin) * sx, (a.ymax - a.ymin) * sy);
  } else if (state.kind === "obb" && det.obb) {
    const o = det.obb;
    octx.save();
    octx.translate(o.cx * sx, o.cy * sy);
    octx.rotate(o.theta);
    octx.strokeRect((-o.width / 2) * sx, (-o.height / 2) * sy, o.width * sx, o.height * sy);
    octx.restore();
  } else if (det.hull) {
    const v = det.hull.vertices;
    if (v.length > 1) {
      octx.beginPath();
      octx.moveTo(v[0][0] * sx, v[0][1] * sy);
      for (let i = 1; i < v.length; i++) octx.lineTo(v[i][0] * sx, v[i][1] * sy);
      octx.closePath();
      octx.stroke();
    }
  }
}

function applyOverlayVis() {
  gtEllGroup.visible = $("showGtEll").checked || $("showGtAx").checked;
  if (gtEllGroup.children[0]) gtEllGroup.children[0].visible = $("showGtEll").checked;
  if (gtEllGroup.children[1]) gtEllGroup.children[1].visible = $("showGtAx").checked;
  estEllGroup.visible = $("showEstEll").checked || $("showEstAx").checked;
  if (estEllGroup.children[0]) estEllGroup.children[0].visible = $("showEstEll").checked;
  if (estEllGroup.children[1]) estEllGroup.children[1].visible = $("showEstAx").checked;
  gtPathLine.visible = $("showGtPath").checked;
  estPathLine.visible = $("showEstPath").checked;
}

$("reset").addEventListener("click", async () => {
  if (solverReachable) {
    await fetch(apiUrl("/api/reset"), { method: "POST" });
  }
  state.estimate = null;
  state.estPath = [];
  state.gtPath = [];
  estEllGroup.clear();
  estPathLine.geometry = new THREE.BufferGeometry();
  gtPathLine.geometry = new THREE.BufferGeometry();
  $("nobs").textContent = "0";
  $("rms").textContent = "—";
  $("status").textContent = "solver reset";
  $("status").className = "ok";
});

function bindSlider(id, key, fmt) {
  const el = $(id);
  const apply = () => {
    state[key] = el.type === "checkbox" ? el.checked : parseFloat(el.value);
    if (fmt) $(id + "V").textContent = fmt(state[key]);
  };
  el.addEventListener("input", apply);
  apply();
}
bindSlider("focal", "focalMm", (v) => `${v} mm`);
bindSlider("baseline", "baseline", (v) => `${v.toFixed(2)} m`);
bindSlider("hz", "hz", (v) => v.toFixed(1));
bindSlider("jitter", "jitter", (v) => `${v} px`);
bindSlider("dnoise", "dnoise", (v) => `${v} px`);
bindSlider("drop", "drop", (v) => `${v}%`);
bindSlider("ddrop", "ddrop", (v) => `${v}%`);
$("drop").addEventListener("input", () => { state.drop = parseFloat($("drop").value) / 100; });
$("ddrop").addEventListener("input", () => { state.ddrop = parseFloat($("ddrop").value) / 100; });
state.drop = 0; state.ddrop = 0;
$("armed").addEventListener("change", () => { state.armed = $("armed").checked; });
$("sensor").addEventListener("change", () => {
  state.sensorMm = parseFloat($("sensor").value);
  applyIntrinsics();
});
document.querySelectorAll('input[name="kind"]').forEach((r) => {
  r.addEventListener("change", () => { if (r.checked) state.kind = r.value; });
});
["showGtEll", "showEstEll", "showGtAx", "showEstAx", "showGtPath", "showEstPath"].forEach((id) => {
  $(id).addEventListener("change", applyOverlayVis);
});

function resize() {
  const side = sensorViewSize();
  renderer.setSize(side, side);
  applyIntrinsics();
  placeOverlayOnViewfinder();
  const mw = miniEl.clientWidth, mh = miniEl.clientHeight;
  miniRenderer.setSize(mw, mh);
  miniCam.aspect = mw / Math.max(mh, 1);
  miniCam.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let last = performance.now();
let lastRateT = performance.now();
let detects = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  fly(dt);
  applyOverlayVis();
  if (state.estimate && state.estimate.initialized && state.estimate.T_cam_from_ell) {
    const TcamEll = mat4FromRowMajor(state.estimate.T_cam_from_ell);
    const M = new THREE.Matrix4().multiplyMatrices(camera.matrixWorld, D_CV);
    M.multiply(TcamEll);
    estEllGroup.matrix.copy(M);
  }
  renderer.render(scene, camera);
  miniRenderer.render(miniScene, miniCam);
  drawDetectionOverlay();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function detectLoop() {
  const period = 1000 / Math.max(state.hz, 0.1);
  if (state.armed) {
    const now = performance.now();
    state.lastDetect = now;
    pushGtPath();
    detects += 1;
    try {
      tickDetect();
    } catch (err) {
      console.warn("detect", err);
    }
    if (now - lastRateT > 1000) {
      $("rate").textContent = detects.toFixed(0);
      detects = 0;
      lastRateT = now;
    }
  }
  setTimeout(detectLoop, period);
}
detectLoop();

fetch(apiUrl("/api/health"))
  .then((r) => {
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  })
  .then(() => {
    solverReachable = true;
    $("status").textContent = "solver ready";
    $("status").className = "ok";
  })
  .catch(() => {
    solverReachable = false;
    $("status").textContent = "fly-only (no solver)";
    $("status").className = "bad";
  });

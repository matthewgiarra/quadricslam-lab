import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { emptyMask, maskSpec, setBit, shiftMask } from "./mask.js";
import { obbFromHull } from "./meas.js";
import { createController, lookQuaternion } from "./trajectory.js";
import { LAB_BOX_SIZE, uniformScaleForLength } from "./model-fit.js";

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
const SCHEMA = "quadricslam.sim.v1";

// Relative URLs so the same files work under FastAPI (`/`) and GitHub Pages.
// Optional `?api=` overrides the stream origin. Live frames need `./run.sh`.
function rewriteLoopbackHost(urlStr) {
  return urlStr
    .replace("://0.0.0.0", "://127.0.0.1")
    .replace("://[::]", "://127.0.0.1")
    .replace("://::", "://127.0.0.1");
}
const API_BASE = (() => {
  const q = new URLSearchParams(location.search).get("api");
  if (q) return rewriteLoopbackHost(q.replace(/\/$/, ""));
  if (location.hostname === "0.0.0.0" || location.hostname === "[::]") {
    return rewriteLoopbackHost(`${location.protocol}//127.0.0.1:${location.port || "8765"}`);
  }
  return "";
})();

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
  recording: false,
  modelLength: 0.8,
  lastDetect: 0,
  gtPath: [],
  lastDet: null,
  seq: 0,
};

const FAR_M = 5_000_000; // 5000 km. Logarithmic depth keeps the near objects stable.
const HOME_POS = [2.6, 0.55, 0.4];

const viewEl = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(viewEl.clientWidth, viewEl.clientHeight);
renderer.setClearColor(0x0b0d12, 1);
viewEl.appendChild(renderer.domElement);

const ktx2Loader = new KTX2Loader();
ktx2Loader.setTranscoderPath(`${THREE_CDN}/examples/jsm/libs/basis/`);
ktx2Loader.detectSupport(renderer);
gltfLoader.setKTX2Loader(ktx2Loader);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(30, 1, 0.02, FAR_M);
camera.position.set(HOME_POS[0], HOME_POS[1], HOME_POS[2]);
camera.lookAt(0, 0, 0);

const miniEl = $("minimap");
const miniRenderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
miniRenderer.setPixelRatio(1);
miniRenderer.setClearColor(0x0a0c10, 1);
miniEl.appendChild(miniRenderer.domElement);
const miniScene = new THREE.Scene();
const miniCam = new THREE.PerspectiveCamera(40, 280 / 200, 0.1, FAR_M);
miniCam.position.set(4.2, 3.4, 4.2);
miniCam.lookAt(0, 0, 0);

function applyIntrinsics() {
  const fov = THREE.MathUtils.radToDeg(2 * Math.atan((state.sensorMm / 2) / state.focalMm));
  camera.fov = fov;
  camera.aspect = 1;
  camera.updateProjectionMatrix();
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
function buildBoxMesh(sx = LAB_BOX_SIZE[0], sy = LAB_BOX_SIZE[1], sz = LAB_BOX_SIZE[2]) {
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

function applyModelLength(root) {
  const natural = root && root.userData.naturalMax;
  if (!natural || !(state.modelLength > 0)) return;
  const scale = uniformScaleForLength(natural, state.modelLength);
  if (scale == null) return;
  root.scale.copy(root.userData.baseScale).multiplyScalar(scale);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  box.getSize(size);
  root.userData.gtRadii = [size.x / 2, size.y / 2, size.z / 2];
  if (subject === root) {
    rebuildGtOverlay();
    detDirty = true;
  }
}

function refreshStreamStatus() {
  const on = ws && ws.readyState === WebSocket.OPEN;
  $("status").textContent = on ? "stream connected" : "stream disconnected";
  $("status").className = on ? "ok" : "bad";
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
    root.userData.naturalMax = Math.max(size.x, size.y, size.z) || 1;
    root.userData.baseScale = root.scale.clone();
    state.modelLength = parseFloat($("glbLength").value);
    if (!(state.modelLength > 0)) state.modelLength = 0.8;
    loadedGlb = root;
    applyModelLength(root);
    $("glbLength").disabled = false;
    $("primitive").value = "glb";
    replaceSubject(root);
    $("status").textContent = `loaded ${file.name}`;
    $("status").className = "ok";
    setTimeout(refreshStreamStatus, 1200);
  } catch (err) {
    console.error(err);
    $("status").textContent = "GLB load failed";
    $("status").className = "bad";
  } finally {
    URL.revokeObjectURL(url);
  }
});

function wireEllipsoid(radii, color) {
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
scene.add(gtEllGroup);

function rebuildGtOverlay() {
  gtEllGroup.clear();
  const r = gtRadii();
  gtEllGroup.add(wireEllipsoid(r, 0x4ade80));
  gtEllGroup.add(axisHelper(r, 1));
}
rebuildGtOverlay();

const overlay = document.createElement("canvas");
overlay.id = "det-overlay";
document.body.appendChild(overlay);
const octx = overlay.getContext("2d");
const maskCanvas = document.createElement("canvas");
maskCanvas.width = DET;
maskCanvas.height = DET;
const maskCtx = maskCanvas.getContext("2d");
const maskImage = maskCtx.createImageData(DET, DET);
let paintedDet = null;

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
  paintedDet = null;
}

const vel = new THREE.Vector3();
const keys = new Set();
let locked = false;
const euler = new THREE.Euler(0, 0, 0, "YXZ");
euler.setFromQuaternion(camera.quaternion);

const traj = createController();
let trajSnap = traj.snapshot();
let flightVel = [0, 0, 0];

function camArr() {
  return [camera.position.x, camera.position.y, camera.position.z];
}
function camQuat() {
  return [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w];
}
function syncObject() {
  traj.params.objectPosition = [objectRoot.position.x, objectRoot.position.y, objectRoot.position.z];
}
function rearmIfCanned() {
  if (traj.params.mode !== "canned") return;
  syncObject();
  traj.onParamsChanged(performance.now() / 1000, camArr(), flightVel.slice(), camQuat());
}
function enterManual() {
  traj.cancel();
  vel.set(flightVel[0], flightVel[1], flightVel[2]);
  euler.setFromQuaternion(camera.quaternion);
  trajSnap = traj.snapshot();
}
function enterCanned() {
  syncObject();
  traj.play(performance.now() / 1000, camArr(), flightVel.slice(), camQuat());
}

viewEl.addEventListener("click", () => renderer.domElement.requestPointerLock());
document.addEventListener("pointerlockchange", () => {
  locked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener("mousemove", (e) => {
  if (!locked || traj.params.mode === "canned") return;
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
  flightVel = [vel.x, vel.y, vel.z];
}

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

function collectMask(needPts, needMask) {
  const step = needPts || needMask ? 1 : 2;
  const pts = [];
  let xmin = DET;
  let ymin = DET;
  let xmax = -1;
  let ymax = -1;
  let hit = 0;
  const bytes = needMask ? emptyMask(DET, DET) : null;
  for (let y = 0; y < DET; y += step) {
    const srcY = DET - 1 - y;
    for (let x = 0; x < DET; x += step) {
      const i = (srcY * DET + x) * 4;
      if (pixelBuf[i] > 20) {
        hit++;
        if (x < xmin) xmin = x;
        if (y < ymin) ymin = y;
        if (x > xmax) xmax = x;
        if (y > ymax) ymax = y;
        if (needPts) pts.push(x * DET_SCALE, y * DET_SCALE);
        if (bytes) setBit(bytes, DET, x, y);
      }
    }
  }
  if (hit < 4) return null;
  return {
    pts,
    xmin: xmin * DET_SCALE,
    ymin: ymin * DET_SCALE,
    xmax: (needMask ? xmax + 1 : xmax) * DET_SCALE,
    ymax: (needMask ? ymax + 1 : ymax) * DET_SCALE,
    mask: bytes ? maskSpec(bytes, DET, DET, DET_SCALE) : null,
    bytes,
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
  if (kind === "aabb") return { aabb, obb: null, hull: null };
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

function renderSilhouetteAt(cam, needPts, needMask) {
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
  renderer.readRenderTargetPixels(detTarget, 0, 0, DET, DET, pixelBuf);
  return collectMask(needPts, needMask);
}

function measurement(kind, feat) {
  if (kind === "mask") return { mask: feat.mask };
  if (kind === "obb") return { obb: feat.obb };
  if (kind === "hull") return { hull: feat.hull };
  return { aabb: feat.aabb };
}

function detShift(dxImg, dyImg) {
  return [Math.round(dxImg / DET_SCALE), Math.round(dyImg / DET_SCALE)];
}

function rowsFromMatrix(m) {
  const e = m.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

function cameraPoseCV() {
  camera.updateMatrixWorld();
  const T = new THREE.Matrix4().multiplyMatrices(camera.matrixWorld, D_CV);
  return rowsFromMatrix(T);
}

function objectPose() {
  objectRoot.updateMatrixWorld();
  return rowsFromMatrix(objectRoot.matrixWorld);
}

function quatToR(q) {
  const x = q.x, y = q.y, z = q.z, w = q.w;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
  ];
}

function ellipsoidGT() {
  if ($("primitive").value !== "ellipsoid") return null;
  const r = gtRadii();
  return {
    radii: [r[0], r[1], r[2]],
    R: quatToR(objectRoot.quaternion),
    t: [objectRoot.position.x, objectRoot.position.y, objectRoot.position.z],
  };
}

let ws = null;
let wsQueue = [];
let reconnectTimer = null;
const recordBuf = [];
const RECORD_CAP = 4000;

function streamUrl() {
  let origin = API_BASE;
  if (!origin) {
    origin = location.protocol === "file:"
      ? "http://127.0.0.1:8765"
      : `${location.protocol}//${location.host}`;
  }
  const u = new URL(rewriteLoopbackHost(origin));
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = "/ws/stream";
  u.search = "";
  u.hash = "";
  return u.toString();
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectStream();
  }, 1000);
}

function connectStream() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  let sock;
  try {
    sock = new WebSocket(streamUrl());
  } catch (err) {
    console.warn("stream", err);
    refreshStreamStatus();
    scheduleReconnect();
    return;
  }
  ws = sock;
  sock.onopen = () => {
    refreshStreamStatus();
    sock.send(JSON.stringify({ type: "hello", role: "publisher" }));
    const queued = wsQueue;
    wsQueue = [];
    for (const text of queued) sock.send(text);
  };
  sock.onclose = () => {
    if (ws === sock) ws = null;
    refreshStreamStatus();
    scheduleReconnect();
  };
  sock.onerror = () => {
    try { sock.close(); } catch { /* close handler reconnects */ }
  };
}

function publishFrame(frame) {
  const text = JSON.stringify(frame);
  if (state.recording) {
    recordBuf.push(text);
    if (recordBuf.length > RECORD_CAP) recordBuf.shift();
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(text);
  } else {
    wsQueue.push(text);
    if (wsQueue.length > 30) wsQueue.shift();
  }
}

function tickDetect() {
  if (!state.armed) return;
  if (Math.random() < state.drop) return;
  syncDetScene();
  const rng = mulRng((performance.now() * 1000) | 0);
  const kind = state.kind;
  const needPts = kind === "obb" || kind === "hull";
  const needMask = kind === "mask";

  const saved = camera.position.clone();
  let maskL = null;
  let maskR = null;
  try {
    maskL = renderSilhouetteAt(camera, needPts, needMask);
    if (maskL) {
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      camera.position.addScaledVector(right, state.baseline);
      camera.updateMatrixWorld();
      maskR = renderSilhouetteAt(camera, needPts, needMask) || maskL;
    }
  } finally {
    camera.position.copy(saved);
    camera.updateMatrixWorld();
  }
  if (!maskL || !maskR) return;

  let fL;
  let fR;
  if (kind === "mask") {
    const jx = () => (state.jitter > 0 ? (rng() * 2 - 1) * state.jitter : 0);
    const jy = () => (state.jitter > 0 ? (rng() * 2 - 1) * state.jitter : 0);
    const [dxL, dyL] = detShift(jx(), jy());
    const dd = state.dnoise > 0 ? (rng() * 2 - 1) * state.dnoise : 0;
    const [dxR, dyR] = detShift(jx() - dd, jy());
    if (dxL || dyL) maskL.bytes = shiftMask(maskL.bytes, DET, DET, dxL, dyL);
    if (dxR || dyR) maskR.bytes = shiftMask(maskR.bytes, DET, DET, dxR, dyR);
    maskL.mask = maskSpec(maskL.bytes, DET, DET, DET_SCALE);
    maskR.mask = maskSpec(maskR.bytes, DET, DET, DET_SCALE);
    fL = { mask: maskL.mask };
    fR = { mask: maskR.mask };
  } else {
    fL = featuresFromMask(maskL, rng, kind);
    fR = featuresFromMask(maskR, rng, kind);
    if (state.dnoise > 0) {
      const dd = (rng() * 2 - 1) * state.dnoise;
      fR.aabb.xmin -= dd;
      fR.aabb.xmax -= dd;
      if (fR.obb) fR.obb.cx -= dd;
      if (fR.hull && fR.hull.vertices.length) {
        fR.hull.vertices = fR.hull.vertices.map((p) => [p[0] - dd, p[1]]);
      }
    }
  }
  // Overlay reads this immediately. Publishing does not wait on a consumer.
  state.lastDet = { kind, left: fL, right: fR, maskBytes: kind === "mask" ? maskL.bytes : null };

  const left = measurement(kind, fL);
  const right = measurement(kind, fR);
  const frame = {
    schema: SCHEMA,
    seq: state.seq,
    t: performance.now() / 1000,
    image_size: [IMAGE, IMAGE],
    K: K(),
    baseline_m: state.baseline,
    kind,
    meas: { left, right },
    pixel_sigma: Math.max(state.jitter, 1.5),
    disparity_sigma: Math.max(state.dnoise, 1.5),
    has_depth: Math.random() >= state.ddrop,
    gt: {
      T_world_from_cam: cameraPoseCV(),
      T_world_from_object: objectPose(),
    },
    traj: trajSnap,
  };
  const ell = ellipsoidGT();
  if (ell) frame.gt.ellipsoid = ell;
  state.seq += 1;
  $("nobs").textContent = String(state.seq);
  publishFrame(frame);
}

const gtPathLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x4ade80 }),
);
miniScene.add(gtPathLine);
const miniSubject = new THREE.Mesh(
  new THREE.SphereGeometry(0.08, 12, 8),
  new THREE.MeshBasicMaterial({ color: 0xc4b5fd }),
);
miniScene.add(miniSubject);
const miniCamDot = new THREE.Mesh(
  new THREE.SphereGeometry(0.06, 10, 8),
  new THREE.MeshBasicMaterial({ color: 0xfbbf24 }),
);
miniScene.add(miniCamDot);

function fitMiniCam() {
  const pts = state.gtPath;
  if (!pts.length) return;
  const box = new THREE.Box3().setFromPoints(pts);
  box.expandByScalar(0.4);
  const c = new THREE.Vector3();
  const s = new THREE.Vector3();
  box.getCenter(c);
  box.getSize(s);
  const span = Math.max(s.length(), 1.2);
  miniCam.position.set(c.x + span * 0.7, c.y + span * 0.55, c.z + span * 0.7);
  miniCam.lookAt(c);
  miniCam.updateProjectionMatrix();
}

function pushGtPath() {
  const p = camera.position.clone();
  state.gtPath.push(p);
  if (state.gtPath.length > 600) state.gtPath.shift();
  miniCamDot.position.copy(p);
  if (state.gtPath.length >= 2) {
    gtPathLine.geometry.dispose();
    gtPathLine.geometry = new THREE.BufferGeometry().setFromPoints(state.gtPath);
  }
  fitMiniCam();
}

function paintMask(bytes) {
  const d = maskImage.data;
  d.fill(0);
  for (let i = 0; i < DET * DET; i++) {
    if (((bytes[i >> 3] >> (i & 7)) & 1) === 0) continue;
    const o = i * 4;
    d[o] = 251;
    d[o + 1] = 191;
    d[o + 2] = 36;
    d[o + 3] = 110;
  }
  maskCtx.putImageData(maskImage, 0, 0);
}

function drawDetectionOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!$("showDet").checked || !state.lastDet) return;
  const det = state.lastDet;
  if (det.kind === "mask" && det.maskBytes) {
    if (paintedDet !== det) {
      paintMask(det.maskBytes);
      paintedDet = det;
    }
    octx.drawImage(maskCanvas, 0, 0, overlay.width, overlay.height);
    return;
  }
  const sx = overlay.width / IMAGE;
  const sy = overlay.height / IMAGE;
  const left = det.left;
  octx.strokeStyle = "rgba(251,191,36,0.95)";
  octx.lineWidth = 1.5;
  if (det.kind === "aabb" && left.aabb) {
    const a = left.aabb;
    octx.strokeRect(a.xmin * sx, a.ymin * sy, (a.xmax - a.xmin) * sx, (a.ymax - a.ymin) * sy);
  } else if (det.kind === "obb" && left.obb) {
    const o = left.obb;
    octx.save();
    octx.translate(o.cx * sx, o.cy * sy);
    octx.rotate(o.theta);
    octx.strokeRect((-o.width / 2) * sx, (-o.height / 2) * sy, o.width * sx, o.height * sy);
    octx.restore();
  } else if (left.hull) {
    const v = left.hull.vertices;
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
  gtPathLine.visible = $("showGtPath").checked;
}

function bindSlider(id, key, fmt) {
  const el = $(id);
  const apply = () => {
    state[key] = el.type === "checkbox" ? el.checked : parseFloat(el.value);
    if (fmt) $(id + "V").textContent = fmt(state[key]);
  };
  el.addEventListener("input", apply);
  apply();
}
bindSlider("hz", "hz", (v) => v.toFixed(1));
bindSlider("jitter", "jitter", (v) => `${v} px`);
bindSlider("dnoise", "dnoise", (v) => `${v} px`);
bindSlider("drop", "drop", (v) => `${v}%`);
bindSlider("ddrop", "ddrop", (v) => `${v}%`);
$("drop").addEventListener("input", () => { state.drop = parseFloat($("drop").value) / 100; });
$("ddrop").addEventListener("input", () => { state.ddrop = parseFloat($("ddrop").value) / 100; });
state.drop = 0;
state.ddrop = 0;
$("armed").addEventListener("change", () => { state.armed = $("armed").checked; });
$("sensor").addEventListener("change", () => {
  state.sensorMm = parseFloat($("sensor").value);
  applyIntrinsics();
});
document.querySelectorAll('input[name="kind"]').forEach((r) => {
  r.addEventListener("change", () => { if (r.checked) state.kind = r.value; });
});
["showGtEll", "showGtAx", "showGtPath"].forEach((id) => {
  $(id).addEventListener("change", applyOverlayVis);
});

function bindTunable(id, { positive = false, ecc = false, apply }) {
  const range = $(id);
  const num = $(id + "N");
  const minEl = $(id + "Min");
  const maxEl = $(id + "Max");
  const decimals = (String(range.step).split(".")[1] || "").length;
  const show = (v) => {
    num.value = decimals ? Number(v).toFixed(decimals) : String(Math.round(v));
    range.value = String(v);
  };
  const commit = (raw) => {
    let v = parseFloat(raw);
    if (!Number.isFinite(v)) return;
    if (positive && !(v > 0)) return;
    if (ecc) v = Math.min(0.949999, Math.max(0, v));
    let lo = parseFloat(minEl.value);
    let hi = parseFloat(maxEl.value);
    if (Number.isFinite(lo) && Number.isFinite(hi)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      if (lo < hi) {
        minEl.value = decimals ? lo.toFixed(decimals) : String(lo);
        maxEl.value = decimals ? hi.toFixed(decimals) : String(hi);
        range.min = String(lo);
        range.max = String(hi);
      }
    }
    show(v);
    apply(v);
  };
  const ends = () => {
    const lo = parseFloat(minEl.value);
    const hi = parseFloat(maxEl.value);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(lo < hi)) return;
    range.min = String(lo);
    range.max = String(hi);
  };
  range.addEventListener("input", () => commit(range.value));
  num.addEventListener("input", () => commit(num.value));
  minEl.addEventListener("change", ends);
  maxEl.addEventListener("change", ends);
  commit(range.value);
}
bindTunable("focal", {
  positive: true,
  apply: (v) => {
    state.focalMm = v;
    applyIntrinsics();
  },
});
bindTunable("baseline", { positive: true, apply: (v) => { state.baseline = v; } });
bindTunable("period", { positive: true, apply: (v) => { traj.params.period = v; rearmIfCanned(); } });
bindTunable("semiMajor", { positive: true, apply: (v) => { traj.params.a = v; rearmIfCanned(); } });
bindTunable("ecc", { ecc: true, apply: (v) => { traj.params.e = v; rearmIfCanned(); } });
bindTunable("offX", { apply: (v) => { traj.params.offset[0] = v; rearmIfCanned(); } });
bindTunable("offY", { apply: (v) => { traj.params.offset[1] = v; rearmIfCanned(); } });
bindTunable("offZ", { apply: (v) => { traj.params.offset[2] = v; rearmIfCanned(); } });
bindTunable("ingress", { positive: true, apply: (v) => { traj.params.ingress = v; rearmIfCanned(); } });

function bindTraj(id, assign, fmt) {
  const el = $(id);
  const apply = () => {
    const v = parseFloat(el.value);
    assign(v);
    if (fmt) $(id + "V").textContent = fmt(v);
    rearmIfCanned();
  };
  el.addEventListener("input", apply);
  apply();
}
bindTraj("incl", (v) => { traj.params.inclDeg = v; }, (v) => `${v.toFixed(0)}°`);
bindTraj("asc", (v) => { traj.params.ascDeg = v; }, (v) => `${v.toFixed(0)}°`);
$("glbLength").addEventListener("input", () => {
  const v = parseFloat($("glbLength").value);
  if (!(v > 0)) return;
  state.modelLength = v;
  if (loadedGlb) applyModelLength(loadedGlb);
});
function resetCameraView() {
  $("trajMode").value = "manual";
  traj.cancel();
  camera.position.set(HOME_POS[0], HOME_POS[1], HOME_POS[2]);
  camera.up.set(0, 1, 0);
  camera.lookAt(0, 0, 0);
  euler.setFromQuaternion(camera.quaternion);
  vel.set(0, 0, 0);
  flightVel = [0, 0, 0];
  trajSnap = traj.snapshot();
}
$("resetView").addEventListener("click", resetCameraView);
$("trajMode").addEventListener("change", () => {
  if ($("trajMode").value === "canned") enterCanned();
  else enterManual();
});
$("trajPlay").addEventListener("click", () => {
  $("trajMode").value = "canned";
  enterCanned();
});
$("trajPause").addEventListener("click", () => {
  traj.pause(performance.now() / 1000);
});
$("trajReset").addEventListener("click", () => {
  $("trajMode").value = "canned";
  syncObject();
  traj.resetPhase(performance.now() / 1000, camArr(), flightVel.slice(), camQuat());
});
document.querySelectorAll('input[name="look"]').forEach((r) => {
  r.addEventListener("change", () => {
    if (!r.checked) return;
    traj.params.look = r.value;
    rearmIfCanned();
  });
});
$("record").addEventListener("change", () => {
  state.recording = $("record").checked;
  if (state.recording) recordBuf.length = 0;
});
$("downloadNdjson").addEventListener("click", () => {
  const text = recordBuf.join("\n") + (recordBuf.length ? "\n" : "");
  const blob = new Blob([text], { type: "application/x-ndjson" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "quadricslam-sim.ndjson";
  a.click();
  URL.revokeObjectURL(a.href);
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

function applyCannedPose(pose) {
  if (!pose || !pose.position) return;
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  const q = pose.quaternion;
  camera.quaternion.set(q[0], q[1], q[2], q[3]);
  euler.setFromQuaternion(camera.quaternion);
  if (pose.velocity) flightVel = pose.velocity.slice();
}

let last = performance.now();
let lastRateT = performance.now();
let detects = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  syncObject();
  if (traj.params.mode === "canned") {
    const pose = traj.sample(now / 1000);
    trajSnap = pose.traj;
    applyCannedPose(pose);
  } else {
    fly(dt);
    trajSnap = traj.snapshot();
  }
  applyOverlayVis();
  renderer.render(scene, camera);
  miniRenderer.render(miniScene, miniCam);
  drawDetectionOverlay();
  requestAnimationFrame(frame);
}
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
function selfCheckFrames() {
  const dummy = new THREE.PerspectiveCamera();
  dummy.position.set(1, 2, 3);
  dummy.up.set(0, 1, 0);
  dummy.lookAt(0, 0, 0);
  dummy.updateMatrixWorld(true);
  const q = lookQuaternion([1, 2, 3], [0, 0, 0], [0, 1, 0]);
  const got = dummy.quaternion.toArray();
  let align = 0;
  for (let i = 0; i < 4; i++) align += q[i] * got[i];
  if (Math.abs(align) < 0.9999) throw new Error(`look quaternion mismatch ${align}`);
  const T = new THREE.Matrix4().multiplyMatrices(dummy.matrixWorld, D_CV);
  const e = T.elements;
  if (Math.abs(e[12] - 1) > 1e-6 || Math.abs(e[13] - 2) > 1e-6 || Math.abs(e[14] - 3) > 1e-6) {
    throw new Error(`OpenCV translation ${e[12]}, ${e[13]}, ${e[14]}`);
  }
  const fwd = new THREE.Vector3(e[8], e[9], e[10]);
  const view = new THREE.Vector3(-1, -2, -3).normalize();
  if (fwd.dot(view) < 0.999) throw new Error(`OpenCV forward ${fwd.toArray()}`);
}
selfCheckFrames();
connectStream();
requestAnimationFrame(frame);
detectLoop();

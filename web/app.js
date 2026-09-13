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

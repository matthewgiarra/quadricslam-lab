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
function rewriteLoopbackHost(urlStr) {
  // 0.0.0.0 / :: are bind addresses. fetch() to them fails in Chrome/Safari.
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
let solverBase = API_BASE;
const apiUrl = (path) => `${solverBase}${path}`;
let solverReachable = false;

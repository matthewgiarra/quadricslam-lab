import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0";
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(`${THREE_CDN}/examples/jsm/libs/draco/gltf/`);
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);
gltfLoader.setMeshoptDecoder(MeshoptDecoder);

const IMAGE = 1024;
const DET = 256;
const DET_SCALE = IMAGE / DET;
const D_CV = new THREE.Matrix4().makeScale(1, -1, -1);

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
let solverBase = API_BASE;
const apiUrl = (path) => `${solverBase}${path}`;
let solverReachable = false;

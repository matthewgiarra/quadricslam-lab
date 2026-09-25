import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { ACE_LENGTH_M, LAB_BOX_SIZE, longestSide, uniformScaleForLength } from "../web/model-fit.js";

function matMul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

function nodeMatrix(node) {
  if (node.matrix) return node.matrix;
  const t = node.translation || [0, 0, 0];
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  const m = [
    1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0,
    2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0,
    2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ];
  m[0] *= s[0]; m[1] *= s[0]; m[2] *= s[0];
  m[4] *= s[1]; m[5] *= s[1]; m[6] *= s[1];
  m[8] *= s[2]; m[9] *= s[2]; m[10] *= s[2];
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
  return m;
}

function parseGlb(data) {
  assert.equal(data.subarray(0, 4).toString(), "glTF");
  let off = 12;
  const chunks = [];
  while (off + 8 <= data.length) {
    const n = data.readUInt32LE(off);
    const typ = data.subarray(off + 4, off + 8).toString();
    chunks.push([typ, data.subarray(off + 8, off + 8 + n)]);
    off += 8 + n;
  }
  const gltf = JSON.parse(chunks[0][1].toString());
  return gltf;
}

function worldSize(gltf) {
  const nodes = gltf.nodes || [];
  const roots = gltf.scenes[gltf.scene].nodes;
  const world = new Map();
  const walk = (i, parent) => {
    const local = nodeMatrix(nodes[i]);
    world.set(i, parent ? matMul(parent, local) : local);
    for (const c of nodes[i].children || []) walk(c, world.get(i));
  };
  for (const i of roots) walk(i, null);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const xf = (m, p) => [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
  for (let i = 0; i < nodes.length; i++) {
    const meshIndex = nodes[i].mesh;
    if (meshIndex == null || !world.has(i)) continue;
    const m = world.get(i);
    for (const prim of gltf.meshes[meshIndex].primitives) {
      const acc = gltf.accessors[prim.attributes.POSITION];
      const a = acc.min;
      const b = acc.max;
      for (const x of [a[0], b[0]]) {
        for (const y of [a[1], b[1]]) {
          for (const z of [a[2], b[2]]) {
            const p = xf(m, [x, y, z]);
            for (let k = 0; k < 3; k++) {
              min[k] = Math.min(min[k], p[k]);
              max[k] = Math.max(max[k], p[k]);
            }
          }
        }
      }
    }
  }
  return [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
}

function cubeGlb(side) {
  const h = side / 2;
  const corners = [
    [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
    [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
  ];
  const faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3], [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]];
  const flat = [];
  for (const f of faces) for (const i of f) flat.push(...corners[i]);
  const bin = Buffer.alloc(flat.length * 4);
  flat.forEach((v, i) => bin.writeFloatLE(v, i * 4));
  const gltf = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{
      bufferView: 0, componentType: 5126, count: flat.length / 3, type: "VEC3",
      min: [-h, -h, -h], max: [h, h, h],
    }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: bin.length }],
    buffers: [{ byteLength: bin.length }],
  };
  let js = Buffer.from(JSON.stringify(gltf));
  while (js.length % 4) js = Buffer.concat([js, Buffer.from(" ")]);
  const chunk = (tag, body) => {
    const head = Buffer.alloc(8);
    head.writeUInt32LE(body.length, 0);
    head.write(tag, 4, 4, "ascii");
    return Buffer.concat([head, body]);
  };
  const body = Buffer.concat([chunk("JSON", js), chunk("BIN\0", bin)]);
  const head = Buffer.alloc(12);
  head.write("glTF", 0, 4, "ascii");
  head.writeUInt32LE(2, 4);
  head.writeUInt32LE(12 + body.length, 8);
  return Buffer.concat([head, body]);
}

test("a unit cube glb loads with longest side 1", () => {
  const size = worldSize(parseGlb(cubeGlb(1)));
  assert.ok(Math.abs(longestSide(size) - 1) < 1e-6, size.join(","));
  const scale = uniformScaleForLength(longestSide(size), 4);
  assert.ok(Math.abs(longestSide(size) * scale - 4) < 1e-9);
});

test("the lab box primitive scales from its 0.5 m longest side", () => {
  assert.equal(longestSide(LAB_BOX_SIZE), 0.5);
  assert.equal(uniformScaleForLength(0.5, 2), 4);
  assert.equal(uniformScaleForLength(0, 1), null);
});

test("ace.glb loads and scales to a longest side of 8.3 m", () => {
  const gltf = parseGlb(readFileSync(new URL("../models/ace.glb", import.meta.url)));
  assert.ok(gltf.meshes.length >= 9);
  assert.equal(gltf.extensionsRequired, undefined);
  const natural = longestSide(worldSize(gltf));
  // File bounds: the solar array spans 7.41 m. 8.3 m is the physical length.
  assert.ok(Math.abs(natural - 7.410069227218628) < 1e-6, natural);
  const scale = uniformScaleForLength(natural, ACE_LENGTH_M);
  assert.ok(Math.abs(natural * scale - 8.3) < 1e-9);
  assert.ok(Math.abs(scale - 8.3 / 7.410069227218628) < 1e-9);
});

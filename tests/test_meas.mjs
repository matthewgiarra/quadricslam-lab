import assert from "node:assert/strict";
import test from "node:test";
import { obbFromHull } from "../web/meas.js";

function close(a, b, tol = 1e-6) {
  assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);
}

test("axis-aligned rectangle center is the midpoint", () => {
  const hull = [[100, 200], [180, 200], [180, 260], [100, 260]];
  const obb = obbFromHull(hull);
  close(obb.cx, 140);
  close(obb.cy, 230);
  const sides = [obb.width, obb.height].sort((a, b) => a - b);
  close(sides[0], 60);
  close(sides[1], 80);
});

test("rotated rectangle keeps its center", () => {
  const center = [140, 230];
  const ang = Math.PI / 6;
  const co = Math.cos(ang);
  const si = Math.sin(ang);
  const corners = [[-40, -30], [40, -30], [40, 30], [-40, 30]].map(([x, y]) => [
    center[0] + x * co - y * si,
    center[1] + x * si + y * co,
  ]);
  const obb = obbFromHull(corners);
  close(obb.cx, center[0]);
  close(obb.cy, center[1]);
  const sides = [obb.width, obb.height].sort((a, b) => a - b);
  close(sides[0], 60);
  close(sides[1], 80);
});

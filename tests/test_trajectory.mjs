import assert from "node:assert/strict";
import test from "node:test";
import {
  cameraUp,
  clampE,
  createController,
  ellipsePoint,
  ellipseTangent,
  hermite,
  hermiteVelocity,
  nearestTheta,
  orbitBasis,
} from "../web/trajectory.js";

function close(a, b, tol = 1e-6) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) <= tol, `index ${i}: ${a[i]} vs ${b[i]}`);
  }
}

test("horizontal e=0 orbit is a circle of radius a", () => {
  const { n, u, v } = orbitBasis(0, 0);
  close(n, [0, 1, 0], 1e-12);
  const a = 2.5;
  const center = [0.2, -0.4, 0.5];
  for (let i = 0; i < 16; i++) {
    const th = (i / 16) * Math.PI * 2;
    const p = ellipsePoint(center, u, v, a, a, th);
    const d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    const r = Math.hypot(d[0], d[1], d[2]);
    assert.ok(Math.abs(r - a) < 1e-9, `radius ${r}`);
    const plane = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    assert.ok(Math.abs(plane) < 1e-9, `plane ${plane}`);
  }
});

test("basis stays orthonormal when the normal is +X", () => {
  const { n, u, v } = orbitBasis(90, 90);
  close(n, [1, 0, 0], 1e-12);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  assert.ok(Math.abs(dot(u, n)) < 1e-9);
  assert.ok(Math.abs(dot(v, n)) < 1e-9);
  assert.ok(Math.abs(dot(u, v)) < 1e-9);
  assert.ok(Math.abs(Math.hypot(...u) - 1) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-12);
});

test("nearest point on the ellipse round-trips", () => {
  const { u, v } = orbitBasis(25, 40);
  const a = 3;
  const e = 0.4;
  const b = a * Math.sqrt(1 - e * e);
  const center = [1, -2, 0.5];
  const theta = 1.7;
  const p = ellipsePoint(center, u, v, a, b, theta);
  const found = nearestTheta(center, u, v, a, b, p);
  const q = ellipsePoint(center, u, v, a, b, found);
  close(q, p, 1e-6);
});

test("hermite matches the end position and the start velocity", () => {
  const { u, v } = orbitBasis(10, -20);
  const a = 2;
  const b = 1.5;
  const center = [0, 0, 0];
  const theta = 0.8;
  const p1 = ellipsePoint(center, u, v, a, b, theta);
  const v1 = ellipseTangent(u, v, a, b, theta, 12);
  const p0 = [1.2, 0.4, -0.7];
  const v0 = [0.2, -0.1, 0.4];
  const T = 3;
  close(hermite(p0, v0, p1, v1, T, 1), p1, 1e-9);
  close(hermite(p0, v0, p1, v1, T, 0), p0, 1e-9);
  close(hermiteVelocity(p0, v0, p1, v1, T, 0), v0, 1e-9);
  close(hermiteVelocity(p0, v0, p1, v1, T, 1), v1, 1e-9);
});

test("ingress ends on the ellipse sample and does not teleport", () => {
  const offset = [0.3, -0.2, 0.4];
  const ctrl = createController({
    a: 2.4,
    e: 0,
    period: 16,
    ingress: 2.5,
    offset,
    inclDeg: 15,
    ascDeg: 30,
    objectPosition: [0, 0, 0],
  });
  const p0 = [3.1, 0.8, 0.4];
  const v0 = [0.15, 0.02, -0.05];
  const q0 = [0, 0, 0, 1];
  ctrl.play(0, p0, v0, q0);
  const start = ctrl.sample(0);
  close(start.position, p0, 1e-9);
  close(start.velocity, v0, 1e-9);
  assert.equal(start.traj.mode, "canned");
  assert.equal(start.traj.ingressing, true);
  assert.equal(start.traj.family, "orbit");

  const end = ctrl.sample(2.5);
  const { n, u, v } = orbitBasis(15, 30);
  const center = [-offset[0], -offset[1], -offset[2]];
  const theta = nearestTheta(center, u, v, 2.4, 2.4, p0);
  const p1 = ellipsePoint(center, u, v, 2.4, 2.4, theta);
  close(end.position, p1, 1e-6);
  assert.equal(end.ingressing, false);
  const radial = [
    end.position[0] - center[0],
    end.position[1] - center[1],
    end.position[2] - center[2],
  ];
  const plane = radial[0] * n[0] + radial[1] * n[1] + radial[2] * n[2];
  assert.ok(Math.abs(plane) < 1e-6);
  assert.ok(Math.abs(Math.hypot(...radial) - 2.4) < 1e-6);
});

test("pause holds the pose and resume continues the same clock", () => {
  const make = () => createController({ a: 2, e: 0.2, period: 10, ingress: 1, inclDeg: 0 });
  const moving = make();
  const paused = make();
  const p0 = [2.5, 0.2, 0.2];
  const v0 = [0, 0, 0];
  const q0 = [0, 0, 0, 1];
  moving.play(0, p0, v0, q0);
  paused.play(0, p0, v0, q0);
  const atOne = paused.sample(1).position.slice();
  paused.pause(1);
  close(paused.sample(8).position, atOne, 1e-9);
  paused.play(8, atOne, [0, 0, 0], q0);
  close(paused.sample(9).position, moving.sample(2).position, 1e-6);
});

test("reset phase ingresses to theta 0", () => {
  const ctrl = createController({ a: 1.8, e: 0, period: 8, ingress: 1.2, inclDeg: 0, ascDeg: 0 });
  const p0 = [0.2, 0.4, 1.5];
  ctrl.resetPhase(0, p0, [0, 0, 0], [0, 0, 0, 1]);
  const end = ctrl.sample(1.2);
  const { u, v } = orbitBasis(0, 0);
  const p1 = ellipsePoint([0, 0, 0], u, v, 1.8, 1.8, 0);
  close(end.position, p1, 1e-6);
});

test("near-horizontal up is world +Y", () => {
  close(cameraUp([0, 1, 0]), [0, 1, 0]);
  close(cameraUp([0, -1, 0]), [0, 1, 0]);
  assert.equal(clampE(2), 0.949999);
  assert.equal(clampE(-1), 0);
});

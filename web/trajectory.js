// Canned orbit in the world frame. No Three.js — the render loop copies
// the sampled position and quaternion onto the camera.

const TAU = Math.PI * 2;

export function clampE(e) {
  if (!Number.isFinite(e)) return 0;
  return Math.min(0.949999, Math.max(0, e));
}

function hypot3(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

function norm3(a) {
  const l = hypot3(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function wrapAngle(th) {
  return ((th % TAU) + TAU) % TAU;
}

export function orbitBasis(inclDeg, ascDeg) {
  const incl = (inclDeg * Math.PI) / 180;
  const asc = (ascDeg * Math.PI) / 180;
  const n = norm3([
    Math.sin(incl) * Math.sin(asc),
    Math.cos(incl),
    Math.sin(incl) * Math.cos(asc),
  ]);
  // Project world +X onto the plane. Fall back to +Y when the normal is
  // nearly +X and that projection vanishes.
  const drop = n[0];
  let u = [1 - n[0] * drop, 0 - n[1] * drop, 0 - n[2] * drop];
  if (hypot3(u) < 1e-8) {
    const dropY = n[1];
    u = [0 - n[0] * dropY, 1 - n[1] * dropY, 0 - n[2] * dropY];
  }
  u = norm3(u);
  const v = cross(n, u);
  return { n, u, v };
}

export function ellipsePoint(center, u, v, a, b, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    center[0] + u[0] * a * c + v[0] * b * s,
    center[1] + u[1] * a * c + v[1] * b * s,
    center[2] + u[2] * a * c + v[2] * b * s,
  ];
}

// World velocity along the ellipse. dθ/dt = 2π / period.
export function ellipseTangent(u, v, a, b, theta, period) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const w = TAU / Math.max(period, 1e-3);
  return [
    (-u[0] * a * s + v[0] * b * c) * w,
    (-u[1] * a * s + v[1] * b * c) * w,
    (-u[2] * a * s + v[2] * b * c) * w,
  ];
}

function dist2(a, b) {
  const d = sub(a, b);
  return dot(d, d);
}

export function nearestTheta(center, u, v, a, b, point) {
  const N = 720;
  const step = TAU / N;
  let bestT = 0;
  let bestD = Infinity;
  for (let i = 0; i < N; i++) {
    const th = i * step;
    const d = dist2(ellipsePoint(center, u, v, a, b, th), point);
    if (d < bestD) {
      bestD = d;
      bestT = th;
    }
  }
  const cost = (th) => dist2(ellipsePoint(center, u, v, a, b, th), point);
  // Polish inside one coarse step. cos/sin are 2π-periodic, so a bracket
  // that crosses 0 is fine.
  const phi = (Math.sqrt(5) - 1) / 2;
  let lo = bestT - step;
  let hi = bestT + step;
  let c = hi - phi * (hi - lo);
  let d = lo + phi * (hi - lo);
  let fc = cost(c);
  let fd = cost(d);
  for (let k = 0; k < 80; k++) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - phi * (hi - lo);
      fc = cost(c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + phi * (hi - lo);
      fd = cost(d);
    }
  }
  return wrapAngle(fc < fd ? c : d);
}

export function hermite(p0, v0, p1, v1, duration, s) {
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = h00 * p0[i] + h10 * v0[i] * duration + h01 * p1[i] + h11 * v1[i] * duration;
  }
  return out;
}

export function hermiteVelocity(p0, v0, p1, v1, duration, s) {
  const s2 = s * s;
  const dh00 = 6 * s2 - 6 * s;
  const dh10 = 3 * s2 - 4 * s + 1;
  const dh01 = -6 * s2 + 6 * s;
  const dh11 = 3 * s2 - 2 * s;
  const out = [0, 0, 0];
  const T = duration || 1e-3;
  for (let i = 0; i < 3; i++) {
    const dpds = dh00 * p0[i] + dh10 * v0[i] * T + dh01 * p1[i] + dh11 * v1[i] * T;
    out[i] = dpds / T;
  }
  return out;
}

// Up hint for the chaser. Near-horizontal orbits use world +Y so a normal
// that has tipped just past the pole does not flip the view upside down.
export function cameraUp(normal) {
  if (Math.abs(normal[1]) > 0.95) return [0, 1, 0];
  return norm3(normal[1] < 0 ? scale(normal, -1) : normal);
}

function quatFromBasis(x, y, z) {
  const m00 = x[0];
  const m01 = y[0];
  const m02 = z[0];
  const m10 = x[1];
  const m11 = y[1];
  const m12 = z[1];
  const m20 = x[2];
  const m21 = y[2];
  const m22 = z[2];
  const trace = m00 + m11 + m22;
  let qw;
  let qx;
  let qy;
  let qz;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
  const l = Math.hypot(qx, qy, qz, qw) || 1;
  return [qx / l, qy / l, qz / l, qw / l];
}

// Camera quaternion, Three.js convention: local +Z points away from `target`
// (the camera looks down −Z). `up` is a hint. Returns [x, y, z, w].
export function lookQuaternion(eye, target, up) {
  let z = sub(eye, target);
  if (hypot3(z) < 1e-12) z = [0, 0, 1];
  z = norm3(z);
  let x = cross(up, z);
  if (hypot3(x) < 1e-8) {
    const fallback = Math.abs(up[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    z = norm3(add(z, scale(fallback, 1e-4)));
    x = cross(fallback, z);
  }
  x = norm3(x);
  const y = cross(z, x);
  return quatFromBasis(x, y, z);
}

export function slerpQuat(a, b, t) {
  let bx = b[0];
  let by = b[1];
  let bz = b[2];
  let bw = b[3];
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) {
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
    cos = -cos;
  }
  if (cos > 0.9995) {
    const x = a[0] + t * (bx - a[0]);
    const y = a[1] + t * (by - a[1]);
    const z = a[2] + t * (bz - a[2]);
    const w = a[3] + t * (bw - a[3]);
    const l = Math.hypot(x, y, z, w) || 1;
    return [x / l, y / l, z / l, w / l];
  }
  const theta = Math.acos(Math.min(1, cos));
  const s = Math.sin(theta);
  const w1 = Math.sin((1 - t) * theta) / s;
  const w2 = Math.sin(t * theta) / s;
  return [
    a[0] * w1 + bx * w2,
    a[1] * w1 + by * w2,
    a[2] * w1 + bz * w2,
    a[3] * w1 + bw * w2,
  ];
}

export function createController(initial = {}) {
  const params = {
    mode: "manual",
    playing: false,
    a: 2.5,
    e: 0,
    offset: [0, 0, 0],
    inclDeg: 0,
    ascDeg: 0,
    period: 20,
    ingress: 3,
    look: "at_object",
    objectPosition: [0, 0, 0],
    ...initial,
  };
  if (initial.offset) params.offset = initial.offset.slice();
  if (initial.objectPosition) params.objectPosition = initial.objectPosition.slice();

  let ingress = null;
  let joinTime = null;
  let thetaLock = 0;
  let pausedAt = null;

  function center() {
    const o = params.objectPosition;
    const off = params.offset;
    // offset = object − center
    return [o[0] - off[0], o[1] - off[1], o[2] - off[2]];
  }

  function geom() {
    const e = clampE(params.e);
    const a = Math.max(params.a, 1e-3);
    const b = a * Math.sqrt(1 - e * e);
    const { n, u, v } = orbitBasis(params.inclDeg, params.ascDeg);
    return { e, a, b, n, u, v, c: center() };
  }

  function arrivalAttitude(p1, theta1, g) {
    const up = cameraUp(g.n);
    let target;
    if (params.look === "tangent") {
      const vel = ellipseTangent(g.u, g.v, g.a, g.b, theta1, params.period);
      const sp = hypot3(vel);
      const dir = sp > 1e-8 ? scale(vel, 1 / sp) : g.u.slice();
      target = add(p1, dir);
    } else {
      target = params.objectPosition.slice();
    }
    return lookQuaternion(p1, target, up);
  }

  function armToward(t, position, velocity, quaternion, theta1) {
    const g = geom();
    const p0 = position.slice();
    const v0 = velocity ? velocity.slice() : [0, 0, 0];
    const p1 = ellipsePoint(g.c, g.u, g.v, g.a, g.b, theta1);
    const v1 = ellipseTangent(g.u, g.v, g.a, g.b, theta1, params.period);
    const T = Math.max(0.05, params.ingress);
    ingress = {
      t0: t,
      T,
      p0,
      v0,
      p1,
      v1,
      theta1,
      q0: quaternion.slice(),
      q1: arrivalAttitude(p1, theta1, g),
    };
    joinTime = null;
    thetaLock = theta1;
    pausedAt = null;
  }

  function arm(t, position, velocity, quaternion) {
    const g = geom();
    const theta1 = nearestTheta(g.c, g.u, g.v, g.a, g.b, position);
    armToward(t, position, velocity, quaternion, theta1);
  }

  function cancel() {
    params.mode = "manual";
    params.playing = false;
    ingress = null;
    joinTime = null;
    pausedAt = null;
  }

  function play(t, position, velocity, quaternion) {
    params.mode = "canned";
    if (pausedAt != null && (ingress || joinTime != null)) {
      const dt = t - pausedAt;
      if (ingress) ingress.t0 += dt;
      if (joinTime != null) joinTime += dt;
      pausedAt = null;
      params.playing = true;
      return;
    }
    if (!ingress && joinTime == null) arm(t, position, velocity, quaternion);
    params.playing = true;
  }

  function pause(t) {
    if (params.mode !== "canned" || pausedAt != null) return;
    pausedAt = t;
    params.playing = false;
  }

  function resetPhase(t, position, velocity, quaternion) {
    params.mode = "canned";
    params.playing = true;
    armToward(t, position, velocity, quaternion, 0);
  }

  function onParamsChanged(t, position, velocity, quaternion) {
    if (params.mode !== "canned") return;
    const wasPlaying = params.playing;
    arm(t, position, velocity, quaternion);
    if (!wasPlaying) {
      pausedAt = t;
      params.playing = false;
    }
  }

  function snapshot(phase, ingressing) {
    return {
      mode: params.mode === "canned" ? "canned" : "manual",
      family: params.mode === "canned" ? "orbit" : null,
      phase: wrapAngle(phase || 0),
      playing: !!params.playing,
      period_s: params.period,
      semi_major_m: params.a,
      eccentricity: clampE(params.e),
      offset_m: params.offset.slice(),
      incl_deg: params.inclDeg,
      asc_deg: params.ascDeg,
      ingress_s: params.ingress,
      look: params.look,
      ingressing: !!ingressing,
    };
  }

  function poseOnEllipse(t, g) {
    const period = Math.max(params.period, 1e-3);
    const th = thetaLock + (TAU * (t - joinTime)) / period;
    const p = ellipsePoint(g.c, g.u, g.v, g.a, g.b, th);
    const velocity = ellipseTangent(g.u, g.v, g.a, g.b, th, period);
    const up = cameraUp(g.n);
    let target;
    if (params.look === "tangent") {
      const sp = hypot3(velocity);
      const dir = sp > 1e-8 ? scale(velocity, 1 / sp) : g.u.slice();
      target = add(p, dir);
    } else {
      target = params.objectPosition.slice();
    }
    return {
      position: p,
      quaternion: lookQuaternion(p, target, up),
      velocity,
      phase: th,
      ingressing: false,
      traj: snapshot(th, false),
    };
  }

  function sample(t) {
    if (params.mode !== "canned") {
      return { traj: snapshot(0, false) };
    }
    const g = geom();
    const tEval = pausedAt != null ? pausedAt : t;
    if (ingress) {
      const s = Math.min(1, Math.max(0, (tEval - ingress.t0) / ingress.T));
      if (s >= 1 && pausedAt == null) {
        joinTime = ingress.t0 + ingress.T;
        thetaLock = ingress.theta1;
        ingress = null;
        return poseOnEllipse(tEval, g);
      }
      return {
        position: hermite(ingress.p0, ingress.v0, ingress.p1, ingress.v1, ingress.T, s),
        quaternion: slerpQuat(ingress.q0, ingress.q1, s),
        velocity: hermiteVelocity(ingress.p0, ingress.v0, ingress.p1, ingress.v1, ingress.T, s),
        phase: ingress.theta1,
        ingressing: true,
        traj: snapshot(ingress.theta1, true),
      };
    }
    if (joinTime == null) return { traj: snapshot(thetaLock, false) };
    return poseOnEllipse(tEval, g);
  }

  return {
    params,
    arm,
    cancel,
    play,
    pause,
    resetPhase,
    onParamsChanged,
    sample,
    snapshot: () => snapshot(thetaLock, ingress != null),
  };
}

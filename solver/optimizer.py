"""Incremental dual-quadric + stereo-rig pose solver.

Gauge: the first left-camera pose is locked at identity (world = first camera).
Scale comes from the stereo baseline (metric).

Hot path is plain NumPy so a later Eigen/C++ port is mechanical:
  pack / unpack state  ->  residuals(state)  ->  LM.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .geometry import (
    aabb_from_dual_conic,
    invert_SE3,
    obb_angle_residual,
    obb_from_dual_conic,
    pose_from_rt,
    pose_from_vec,
    pose_to_vec,
    predicted_conic,
    rotvec_to_R,
    sample_ellipsoid_wire,
    triangulate_center,
)
from .types import (
    AABB,
    CameraIntrinsics,
    ConvexHull,
    Ellipsoid,
    MeasurementType,
    MonoMeasurement,
    OBB,
    SolverEstimate,
    StereoObservation,
    StereoRig,
)


# Cap the number of views that enter the LM so the demo stays interactive
# at high detection rates. Oldest views are subsampled, first view is kept.
MAX_ACTIVE = 48
MAX_HISTORY = 400


def _mono_from_dict(kind: MeasurementType, d: dict) -> MonoMeasurement:
    m = MonoMeasurement(kind=kind)
    if d.get("aabb") is not None:
        m.aabb = AABB.from_dict(d["aabb"])
    if d.get("obb") is not None:
        m.obb = OBB.from_dict(d["obb"])
    if d.get("hull") is not None:
        m.hull = ConvexHull.from_dict(d["hull"])
    # If only one representation was sent, still fill what we can
    if kind == "aabb" and m.aabb is None and "xmin" in d:
        m.aabb = AABB.from_dict(d)
    if kind == "obb" and m.obb is None and "cx" in d:
        m.obb = OBB.from_dict(d)
    if kind == "hull" and m.hull is None and "vertices" in d:
        m.hull = ConvexHull.from_dict(d)
    return m


def observation_from_payload(p: dict) -> StereoObservation:
    kind: MeasurementType = p.get("kind", "aabb")
    K = CameraIntrinsics.from_dict(p["K"])
    rig = StereoRig(baseline=float(p.get("baseline", 0.1)))
    return StereoObservation(
        stamp=float(p.get("stamp", 0.0)),
        K=K,
        rig=rig,
        kind=kind,
        left=_mono_from_dict(kind, p["left"]),
        right=_mono_from_dict(kind, p["right"]),
        pixel_sigma=float(p.get("pixel_sigma", 2.0)),
        disparity_sigma=float(p.get("disparity_sigma", 2.0)),
        has_depth=bool(p.get("has_depth", True)),
    )


def _center_of(m: MonoMeasurement) -> Optional[np.ndarray]:
    if m.obb is not None:
        return np.array([m.obb.cx, m.obb.cy])
    if m.aabb is not None:
        return m.aabb.center()
    if m.hull is not None and len(m.hull.vertices) > 0:
        return m.hull.vertices.mean(axis=0)
    return None


def _size_of(m: MonoMeasurement) -> Optional[tuple[float, float]]:
    if m.obb is not None:
        return m.obb.width, m.obb.height
    if m.aabb is not None:
        return m.aabb.width(), m.aabb.height()
    if m.hull is not None and len(m.hull.vertices) >= 3:
        v = m.hull.vertices
        return float(v[:, 0].ptp()), float(v[:, 1].ptp())
    return None


def _T_right(T_wl: np.ndarray, rig: StereoRig) -> np.ndarray:
    """World-from-right-camera given world-from-left and a rectified rig."""
    return T_wl @ rig.T_right_from_left()


class QuadricSolver:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.observations: list[StereoObservation] = []
        self.poses: list[np.ndarray] = []  # T_world_from_left, pose[0] locked to I
        self.stamps: list[float] = []
        self.ellipsoid: Optional[Ellipsoid] = None
        self.initialized = False
        self.last_rms = 0.0

    # ------------------------------------------------------------------ init
    def _init_from_first(self, obs: StereoObservation) -> bool:
        cL = _center_of(obs.left)
        cR = _center_of(obs.right)
        size = _size_of(obs.left)
        if cL is None or cR is None or size is None:
            return False
        p_left = triangulate_center(cL[0], cL[1], cR[0], obs.K, obs.rig.baseline)
        if p_left is None:
            return False
        w, h = size
        # Approximate an initially camera-aligned ellipsoid from apparent size.
        sx = max(0.5 * w * p_left[2] / obs.K.fx, 1e-3)
        sy = max(0.5 * h * p_left[2] / obs.K.fy, 1e-3)
        sz = 0.5 * (sx + sy)
        T0 = np.eye(4)
        self.poses = [T0]
        self.stamps = [obs.stamp]
        self.observations = [obs]
        self.ellipsoid = Ellipsoid(
            R=np.eye(3),
            t=p_left.copy(),
            radii=np.array([sx, sy, sz], dtype=np.float64),
        )
        self.initialized = True
        return True

    def _init_new_pose(self, obs: StereoObservation) -> np.ndarray:
        """Initialize a new left-camera pose from stereo center + current ellipsoid."""
        assert self.ellipsoid is not None
        cL = _center_of(obs.left)
        cR = _center_of(obs.right)
        prev = self.poses[-1]
        if cL is None or cR is None:
            return prev.copy()
        p_cam = triangulate_center(cL[0], cL[1], cR[0], obs.K, obs.rig.baseline)
        if p_cam is None:
            return prev.copy()
        # We want T_wc such that T_cw @ ellipsoid.center ≈ p_cam
        # T_cw[:3,:3] @ t_e + t_cw = p_cam
        # Re-use previous rotation, solve translation:
        # R_cw_prev @ t_e + t_cw = p_cam  => t_cw = p_cam - R_cw_prev @ t_e
        T_cw_prev = invert_SE3(prev)
        R_cw = T_cw_prev[:3, :3]
        t_cw = p_cam - R_cw @ self.ellipsoid.t
        T_cw = pose_from_rt(R_cw, t_cw)
        return invert_SE3(T_cw)

    # ------------------------------------------------------------- residuals
    def _active_indices(self) -> list[int]:
        n = len(self.observations)
        if n <= MAX_ACTIVE:
            return list(range(n))
        # Always keep the first few (gauge + scale) and a uniform sample + tail
        keep = set(range(min(3, n)))
        tail = min(16, n)
        keep.update(range(n - tail, n))
        remaining = MAX_ACTIVE - len(keep)
        if remaining > 0:
            mid = np.linspace(3, n - tail - 1, remaining, dtype=int) if n - tail > 3 else []
            keep.update(int(i) for i in mid if 0 <= i < n)
        return sorted(keep)

    def _pack(self, active: list[int]) -> np.ndarray:
        assert self.ellipsoid is not None
        parts = [self.ellipsoid.to_vec()]
        # pose 0 is locked at identity; pack the rest of the active set
        for i in active:
            if i == 0:
                continue
            parts.append(pose_to_vec(self.poses[i]))
        return np.concatenate(parts)

    def _unpack(self, x: np.ndarray, active: list[int]) -> tuple[Ellipsoid, dict[int, np.ndarray]]:
        ell = Ellipsoid.from_vec(x[:9])
        poses: dict[int, np.ndarray] = {0: self.poses[0]}
        k = 9
        for i in active:
            if i == 0:
                continue
            poses[i] = pose_from_vec(x[k : k + 6])
            k += 6
        return ell, poses

    def _mono_residuals(
        self,
        kind: MeasurementType,
        meas: MonoMeasurement,
        Cs: np.ndarray,
        pixel_sigma: float,
    ) -> list[float]:
        sig = max(pixel_sigma, 1e-3)
        if kind == "aabb":
            if meas.aabb is None:
                return []
            pred = aabb_from_dual_conic(Cs)
            if pred is None:
                return [50.0 / sig] * 4
            return ((pred.as_array() - meas.aabb.as_array()) / sig).tolist()
        if kind == "obb":
            if meas.obb is None:
                return []
            pred = obb_from_dual_conic(Cs)
            if pred is None:
                return [50.0 / sig] * 5
            r = [
                (pred.cx - meas.obb.cx) / sig,
                (pred.cy - meas.obb.cy) / sig,
                (pred.width - meas.obb.width) / sig,
                (pred.height - meas.obb.height) / sig,
                obb_angle_residual(pred.theta, meas.obb.theta) / (5.0 * np.pi / 180.0),
            ]
            return r
        # hull
        if meas.hull is None or len(meas.hull.vertices) < 3:
            return []
        Cs_n = Cs / (np.linalg.norm(Cs) + 1e-18)
        out = []
        for l in meas.hull.edges_as_lines():
            # Tangency: l^T C* l = 0. Also push the conic to stay inside the hull
            # by penalizing hull vertices that fall inside the primal ellipse.
            out.append(float(l @ Cs_n @ l) / 0.05)
        return out

    def _residuals_for(self, ell: Ellipsoid, T_wl: np.ndarray, obs: StereoObservation) -> list[float]:
        K = obs.K.K()
        T_wr = _T_right(T_wl, obs.rig)
        r: list[float] = []
        CsL = predicted_conic(ell, K, T_wl)
        CsR = predicted_conic(ell, K, T_wr)
        if CsL is None or CsR is None:
            return [20.0] * 8
        r.extend(self._mono_residuals(obs.kind, obs.left, CsL, obs.pixel_sigma))
        r.extend(self._mono_residuals(obs.kind, obs.right, CsR, obs.pixel_sigma))
        if obs.has_depth:
            cL = _center_of(obs.left)
            cR = _center_of(obs.right)
            if cL is not None and cR is not None:
                d_meas = float(cL[0] - cR[0])
                # Predicted disparity of the ellipsoid center
                T_cw = invert_SE3(T_wl)
                p_cam = T_cw[:3, :3] @ ell.t + T_cw[:3, 3]
                if p_cam[2] > 1e-4:
                    d_pred = obs.K.fx * obs.rig.baseline / p_cam[2]
                else:
                    d_pred = 0.0
                r.append((d_pred - d_meas) / max(obs.disparity_sigma, 1e-3))
        # Soft prior: keep radii positive-already via log, and not wildly anisotropic
        # (helps the first few views). Weight dies off as views accumulate.
        return r

    def _fun(self, x: np.ndarray, active: list[int]) -> np.ndarray:
        ell, poses = self._unpack(x, active)
        rs: list[float] = []
        for i in active:
            T = poses.get(i, self.poses[i])
            rs.extend(self._residuals_for(ell, T, self.observations[i]))
        # Weak regularizer on log-radii so a single view cannot explode the shape
        n = max(len(active), 1)
        w = 0.15 / np.sqrt(n)
        log_r = x[6:9]
        rs.extend(((log_r - log_r.mean()) / 0.5 * w).tolist())
        if not rs:
            return np.array([0.0])
        return np.asarray(rs, dtype=np.float64)

    def _refine(self, max_nfev: int = 30) -> None:
        if not self.initialized or self.ellipsoid is None:
            return
        active = self._active_indices()
        x0 = self._pack(active)
        try:
            kwargs = dict(args=(active,), max_nfev=max_nfev, xtol=1e-6, ftol=1e-6, gtol=1e-6)
            try:
                res = least_squares(self._fun, x0, method="lm", **kwargs)
            except ValueError:
                res = least_squares(self._fun, x0, method="trf", loss="soft_l1", f_scale=3.0, **kwargs)
            ell, poses = self._unpack(res.x, active)
            # Keep radii in a sane band
            med = float(np.median(ell.radii))
            ell.radii = np.clip(ell.radii, max(1e-3, 0.08 * med), 50.0)
            self.ellipsoid = ell
            for i, T in poses.items():
                self.poses[i] = T
            self.last_rms = float(np.sqrt(np.mean(res.fun**2))) if res.fun.size else 0.0
        except Exception:
            # Keep the previous estimate if LM fails this tick
            pass

    # ---------------------------------------------------------------- public
    def observe(self, obs: StereoObservation) -> SolverEstimate:
        if not self.initialized:
            if not self._init_from_first(obs):
                return self.estimate()
            # A few extra iterations on the first view (ellipsoid only; pose locked)
            self._refine(max_nfev=20)
            return self.estimate()

        T_new = self._init_new_pose(obs)
        self.observations.append(obs)
        self.poses.append(T_new)
        self.stamps.append(obs.stamp)
        if len(self.observations) > MAX_HISTORY:
            # Drop from the middle, keep first and tail
            drop = 4
            del self.observations[3 : 3 + drop]
            del self.poses[3 : 3 + drop]
            del self.stamps[3 : 3 + drop]
        nfev = 40 if len(self.observations) < 8 else 25
        self._refine(max_nfev=nfev)
        return self.estimate()

    def observe_payload(self, payload: dict) -> SolverEstimate:
        return self.observe(observation_from_payload(payload))

    def estimate(self) -> SolverEstimate:
        ell = self.ellipsoid or Ellipsoid(np.eye(3), np.zeros(3), np.ones(3))
        return SolverEstimate(
            ellipsoid=ell,
            poses=[p.copy() for p in self.poses],
            stamps=list(self.stamps),
            n_observations=len(self.observations),
            rms=self.last_rms,
            initialized=self.initialized,
        )

    def estimate_dict(self) -> dict:
        est = self.estimate()
        path_obj = []
        current_obj = np.eye(4).tolist()
        if est.initialized and est.poses:
            for T_wc in est.poses:
                path_obj.append(est.pose_in_object_frame(T_wc).tolist())
            current_obj = path_obj[-1]
        T_wl = est.poses[-1] if est.poses else np.eye(4)
        T_wo = est.ellipsoid.T()
        # Placement of estimated ellipsoid into a frame whose camera matches
        # the *current estimated* camera — the client then left-multiplies the
        # live GT camera pose.  T_cam_from_ell = inv(T_wc) @ T_wo
        T_cam_from_ell = invert_SE3(T_wl) @ T_wo
        path_centers = [T[:3, 3].tolist() for T in est.poses]
        return {
            "initialized": est.initialized,
            "n_observations": est.n_observations,
            "rms": est.rms,
            "ellipsoid": est.ellipsoid.as_dict(),
            "T_cam_from_ell": T_cam_from_ell.tolist(),
            "T_wc_current": T_wl.tolist(),
            "T_current_in_object": current_obj,
            "path_in_object": path_obj,
            "path_centers_world": path_centers,
            "wire": sample_ellipsoid_wire(est.ellipsoid) if est.initialized else None,
        }

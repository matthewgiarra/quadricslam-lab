"""Projective geometry for dual quadrics / conics.

Conventions
-----------
* OpenCV camera frame: +X right, +Y down, +Z forward.
* A pose T_wc is camera-from-world? NO: T_wc means world-from-camera,
  i.e. p_world = R_wc @ p_cam + t_wc.  This matches Three.js matrixWorld
  after the Y/Z flip.
* The camera matrix is P = K @ [R_cw | t_cw] = K @ T_cw[:3, :].
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .types import AABB, CameraIntrinsics, Ellipsoid, OBB


def rotvec_to_R(rv: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rv, dtype=np.float64)).as_matrix()


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def invert_SE3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def pose_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_from_vec(vec: np.ndarray) -> np.ndarray:
    return pose_from_rt(rotvec_to_R(vec[:3]), vec[3:6])


def pose_to_vec(T: np.ndarray) -> np.ndarray:
    return np.concatenate([R_to_rotvec(T[:3, :3]), T[:3, 3]])


def camera_matrix(K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    T_cw = invert_SE3(T_wc)
    return K @ T_cw[:3, :]


def project_dual_quadric(Q_star: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ Q_star @ P.T


def normalize_conic(C: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(C)
    if n < 1e-18:
        return C
    return C / n


def aabb_from_dual_conic(Cs: np.ndarray) -> AABB | None:
    """Axis-aligned bounds of the projected ellipse via tangent dual lines.

    Vertical tangents l = (1, 0, -x):  Cs00 - 2 Cs02 x + Cs22 x^2 = 0
    Horizontal tangents l = (0, 1, -y): Cs11 - 2 Cs12 y + Cs22 y^2 = 0
    """
    Cs = 0.5 * (Cs + Cs.T)
    nrm = np.linalg.norm(Cs)
    if not np.isfinite(nrm) or nrm < 1e-18:
        return None
    Cs = Cs / nrm
    c11, c12, c13 = Cs[0, 0], Cs[0, 1], Cs[0, 2]
    c22, c23 = Cs[1, 1], Cs[1, 2]
    c33 = Cs[2, 2]

    def _roots(a: float, b: float, c: float) -> np.ndarray | None:
        # a x^2 + b x + c = 0
        if abs(a) < 1e-16:
            if abs(b) < 1e-16:
                return None
            x = -c / b
            return np.array([x, x])
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            if disc > -1e-8:
                disc = 0.0
            else:
                return None
        s = np.sqrt(disc)
        return np.array([(-b - s) / (2 * a), (-b + s) / (2 * a)])

    xs = _roots(c33, -2.0 * c13, c11)
    ys = _roots(c33, -2.0 * c23, c22)
    if xs is None or ys is None:
        return None
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        return None
    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
        return None
    return AABB(xmin, ymin, xmax, ymax)


def ellipse_params_from_dual(Cs: np.ndarray) -> tuple[np.ndarray, float, float, float] | None:
    """Return (center, major, minor, theta) of the primal ellipse, or None."""
    Cs = 0.5 * (Cs + Cs.T)
    try:
        C = np.linalg.inv(Cs)
    except np.linalg.LinAlgError:
        return None
    C = 0.5 * (C + C.T)
    a, b, c = C[0, 0], 2.0 * C[0, 1], C[1, 1]
    d, e, f = 2.0 * C[0, 2], 2.0 * C[1, 2], C[2, 2]
    den = b * b - 4.0 * a * c
    if abs(den) < 1e-16:
        return None
    cx = (2.0 * c * d - b * e) / den
    cy = (2.0 * a * e - b * d) / den
    # Translate to center and extract axes
    # Evaluate quadratic form of the 2x2 block at the affine ellipse
    Aq = np.array([[a, b / 2.0], [b / 2.0, c]])
    # The implicit ellipse at center: X^T Aq X + g = 0 with g such that
    # C_homog on (x,y,1) = 0. Compute rhs from the constant term after shift.
    val = np.array([cx, cy, 1.0]) @ C @ np.array([cx, cy, 1.0])
    # Points on ellipse satisfy X^T Aq X = -val + 2*linear that vanished at center
    # After completing the square, X^T Aq X = -C_full_at_center wait:
    # x^T C x = [X;1]^T C [X;1] = X^T A X + l·X + f
    # at center linear term is 0, so X^T A X = -f_shifted = -val
    if val >= 0:
        # Degenerate or imaginary; try flipping C
        C = -C
        val = np.array([cx, cy, 1.0]) @ C @ np.array([cx, cy, 1.0])
        Aq = -Aq
        if val >= 0:
            return None
    M = Aq / (-val)
    w, V = np.linalg.eigh(M)
    if (w <= 0).any():
        return None
    axes = 1.0 / np.sqrt(w)
    # V columns are eigenvectors; angle of first (smaller eigenvalue = major? eigh ascending)
    # w[0] <= w[1] => axes[0] >= axes[1], so axes[0] is major
    theta = float(np.arctan2(V[1, 0], V[0, 0]))
    return np.array([cx, cy], dtype=np.float64), float(axes[0]), float(axes[1]), theta


def obb_from_dual_conic(Cs: np.ndarray) -> OBB | None:
    p = ellipse_params_from_dual(Cs)
    if p is None:
        return None
    c, major, minor, theta = p
    return OBB(float(c[0]), float(c[1]), 2.0 * major, 2.0 * minor, theta)


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def obb_angle_residual(pred: float, meas: float) -> float:
    """Smallest difference considering 180 deg OBB ambiguity."""
    d = wrap_angle(pred - meas)
    d2 = wrap_angle(d + np.pi)
    return d if abs(d) < abs(d2) else d2


def predicted_conic(ellipsoid: Ellipsoid, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray | None:
    P = camera_matrix(K, T_wc)
    Cs = project_dual_quadric(ellipsoid.dual_matrix(), P)
    if not np.isfinite(Cs).all():
        return None
    return Cs


def triangulate_center(uL: float, vL: float, uR: float, K: CameraIntrinsics, baseline: float) -> np.ndarray | None:
    """Mid-point triangulation of a stereo correspondence, left-camera frame."""
    d = uL - uR
    if abs(d) < 1e-4:
        return None
    Z = K.fx * baseline / d
    if Z <= 1e-4:
        return None
    X = (uL - K.cx) * Z / K.fx
    Y = (vL - K.cy) * Z / K.fy
    return np.array([X, Y, Z], dtype=np.float64)


def sample_ellipsoid_wire(ellipsoid: Ellipsoid, n: int = 64) -> dict:
    """Three principal rings in world coordinates, for overlays."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    rings = []
    for axes in ((0, 1), (0, 2), (1, 2)):
        pts = np.zeros((n, 3))
        pts[:, axes[0]] = ellipsoid.radii[axes[0]] * c
        pts[:, axes[1]] = ellipsoid.radii[axes[1]] * s
        pts = (ellipsoid.R @ pts.T).T + ellipsoid.t
        rings.append(pts.tolist())
    axes_pts = []
    for i in range(3):
        p0 = ellipsoid.t
        p1 = ellipsoid.t + ellipsoid.R[:, i] * ellipsoid.radii[i]
        axes_pts.append([p0.tolist(), p1.tolist()])
    return {"rings": rings, "axes": axes_pts}


def look_at_cv(cam_world: np.ndarray, target_world: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build T_wc (OpenCV) whose +Z looks at target and whose Y is down-ish."""
    if up is None:
        up = np.array([0.0, -1.0, 0.0])  # OpenCV "up" is -Y of a Y-down frame...
        # Use world -Y as OpenCV up if world is Y-down? We'll use +Y world as gravity-up
        up = np.array([0.0, 1.0, 0.0])
    z = target_world - cam_world
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        z = np.array([0.0, 0.0, 1.0])
    else:
        z = z / zn
    # OpenCV: Y should point down. If world +Y is up, camera Y ~ -world Y.
    world_up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, world_up)  # not quite — OpenCV x = y_down × z?
    # R_wc columns are camera axes in world: X_right, Y_down, Z_fwd
    # Z_fwd = z (toward target)
    # X_right = world_up × Z   if world_up is UP, this is right-handed with Y down?
    # X = up × Z  (up = +Y) gives X to the right when looking forward in XZ
    x = np.cross(world_up, z)
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / xn
    y = np.cross(z, x)  # down-ish
    R = np.column_stack([x, y, z])
    return pose_from_rt(R, cam_world)

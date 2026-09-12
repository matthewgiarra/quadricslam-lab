"""Perfect stereo detections of a known ellipsoid along a prescribed path.

Used by unit tests and as a reference for the JS projector. Not required by
the interactive demo (the browser extracts silhouettes from the framebuffer).
"""

from __future__ import annotations

import numpy as np

from .geometry import aabb_from_dual_conic, look_at_cv, predicted_conic
from .optimizer import _T_right
from .types import (
    AABB,
    CameraIntrinsics,
    Ellipsoid,
    MonoMeasurement,
    StereoObservation,
    StereoRig,
)


def circular_orbit_poses(
    n: int,
    radius: float = 3.0,
    height: float = 0.4,
    target: np.ndarray | None = None,
) -> list[np.ndarray]:
    if target is None:
        target = np.zeros(3)
    poses = []
    for i in range(n):
        th = 2.0 * np.pi * i / n
        cam = np.array([radius * np.cos(th), height, radius * np.sin(th)])
        poses.append(look_at_cv(cam, target))
    return poses


def project_aabb(ell: Ellipsoid, K: CameraIntrinsics, T_wc: np.ndarray) -> AABB | None:
    Cs = predicted_conic(ell, K.K(), T_wc)
    if Cs is None:
        return None
    return aabb_from_dual_conic(Cs)


def make_stereo_aabb_obs(
    ell: Ellipsoid,
    T_wl: np.ndarray,
    K: CameraIntrinsics,
    rig: StereoRig,
    stamp: float = 0.0,
    pixel_jitter: float = 0.0,
    disparity_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> StereoObservation | None:
    T_wr = _T_right(T_wl, rig)
    bL = project_aabb(ell, K, T_wl)
    bR = project_aabb(ell, K, T_wr)
    if bL is None or bR is None:
        return None
    if rng is None:
        rng = np.random.default_rng()
    if pixel_jitter > 0:
        jL = rng.normal(0.0, pixel_jitter, size=4)
        jR = rng.normal(0.0, pixel_jitter, size=4)
        bL = AABB(*(bL.as_array() + jL))
        bR = AABB(*(bR.as_array() + jR))
    if disparity_jitter > 0:
        dd = float(rng.normal(0.0, disparity_jitter))
        bR = AABB(bR.xmin - dd, bR.ymin, bR.xmax - dd, bR.ymax)
    return StereoObservation(
        stamp=stamp,
        K=K,
        rig=rig,
        kind="aabb",
        left=MonoMeasurement(kind="aabb", aabb=bL),
        right=MonoMeasurement(kind="aabb", aabb=bR),
        pixel_sigma=max(pixel_jitter, 1.0),
        disparity_sigma=max(disparity_jitter, 1.0),
        has_depth=True,
    )

"""Sanity checks that do not require the web stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver import CameraIntrinsics, Ellipsoid, QuadricSolver, StereoRig
from solver.geometry import invert_SE3
from solver.synthetic import circular_orbit_poses, make_stereo_aabb_obs


def test_perfect_orbit_recovers_center_and_scale():
    K = CameraIntrinsics.from_lens(50.0, 11.313708, 1024, 1024)
    rig = StereoRig(baseline=0.12)
    gt = Ellipsoid(
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.0]),
        radii=np.array([0.35, 0.20, 0.15]),
    )
    poses = circular_orbit_poses(16, radius=2.5, height=0.35, target=gt.t)
    slv = QuadricSolver()
    for i, T in enumerate(poses):
        obs = make_stereo_aabb_obs(gt, T, K, rig, stamp=float(i))
        assert obs is not None, f"projection failed at view {i}"
        slv.observe(obs)
    est = slv.estimate()
    assert est.initialized
    assert est.n_observations == 16
    # Compare in object frame: ellipsoid center vs GT origin expressed through poses
    # Recovered radii should be in the right ballpark (AABB is a loose constraint).
    r = np.sort(est.ellipsoid.radii)
    g = np.sort(gt.radii)
    rel = np.abs(r - g) / g
    print("radii gt", g, "est", r, "rel", rel)
    print("rms", est.rms, "center", est.ellipsoid.t)
    # Path should orbit at ~2.5 m from the estimated object center
    dists = []
    for T_wc in est.poses:
        T_oc = est.pose_in_object_frame(T_wc)
        dists.append(np.linalg.norm(T_oc[:3, 3]))
    print("path radii", float(np.mean(dists)), "±", float(np.std(dists)))
    assert np.mean(rel) < 0.6, f"radii too far off: {rel}"
    assert abs(np.mean(dists) - 2.5) < 1.0


def test_reset_clears_history():
    K = CameraIntrinsics.from_lens()
    rig = StereoRig()
    gt = Ellipsoid(np.eye(3), np.zeros(3), np.array([0.3, 0.3, 0.2]))
    T = circular_orbit_poses(1, radius=2.0)[0]
    slv = QuadricSolver()
    slv.observe(make_stereo_aabb_obs(gt, T, K, rig, stamp=0.0))
    assert slv.initialized
    slv.reset()
    assert not slv.initialized
    assert slv.observations == []


if __name__ == "__main__":
    test_perfect_orbit_recovers_center_and_scale()
    test_reset_clears_history()
    print("ok")

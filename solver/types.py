from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


MeasurementType = Literal["aabb", "obb", "hull"]


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1024
    height: int = 1024

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def as_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_dict(d: dict) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d.get("width", 1024)),
            height=int(d.get("height", 1024)),
        )

    @staticmethod
    def from_lens(
        focal_mm: float = 50.0,
        sensor_mm: float = 11.313708,  # 1" format diagonal 16mm, square crop
        width: int = 1024,
        height: int = 1024,
    ) -> "CameraIntrinsics":
        fx = focal_mm / sensor_mm * width
        fy = focal_mm / sensor_mm * height
        return CameraIntrinsics(fx, fy, width / 2.0, height / 2.0, width, height)


@dataclass
class StereoRig:
    """Rectified horizontal stereo. Baseline in metres along +X (OpenCV)."""

    baseline: float = 0.10

    def T_left_from_right(self) -> np.ndarray:
        T = np.eye(4)
        T[0, 3] = -self.baseline
        return T

    def T_right_from_left(self) -> np.ndarray:
        T = np.eye(4)
        T[0, 3] = self.baseline
        return T


@dataclass
class AABB:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def as_array(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float64)

    def center(self) -> np.ndarray:
        return np.array(
            [(self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5],
            dtype=np.float64,
        )

    def width(self) -> float:
        return float(self.xmax - self.xmin)

    def height(self) -> float:
        return float(self.ymax - self.ymin)

    @staticmethod
    def from_dict(d: dict) -> "AABB":
        return AABB(float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"]))


@dataclass
class OBB:
    cx: float
    cy: float
    width: float
    height: float
    theta: float  # radians, rotation of the box x-axis

    def as_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.width, self.height, self.theta], dtype=np.float64)

    @staticmethod
    def from_dict(d: dict) -> "OBB":
        return OBB(
            float(d["cx"]),
            float(d["cy"]),
            float(d["width"]),
            float(d["height"]),
            float(d["theta"]),
        )


@dataclass
class ConvexHull:
    """Image-space convex hull vertices, shape (N, 2), clockwise or ccw."""

    vertices: np.ndarray

    def edges_as_lines(self) -> np.ndarray:
        """Return Nx3 homogeneous line coefficients, normalized so a^2+b^2=1."""
        v = self.vertices
        n = len(v)
        lines = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            p = np.array([v[i, 0], v[i, 1], 1.0])
            q = np.array([v[(i + 1) % n, 0], v[(i + 1) % n, 1], 1.0])
            l = np.cross(p, q)
            nrm = np.hypot(l[0], l[1])
            if nrm < 1e-12:
                continue
            lines[i] = l / nrm
        return lines

    @staticmethod
    def from_dict(d: dict) -> "ConvexHull":
        verts = np.asarray(d["vertices"], dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("hull vertices must be Nx2")
        return ConvexHull(verts)


@dataclass
class MonoMeasurement:
    kind: MeasurementType
    aabb: Optional[AABB] = None
    obb: Optional[OBB] = None
    hull: Optional[ConvexHull] = None


@dataclass
class StereoObservation:
    stamp: float
    K: CameraIntrinsics
    rig: StereoRig
    kind: MeasurementType
    left: MonoMeasurement
    right: MonoMeasurement
    # Optional per-observation noise scales used as residual weights
    pixel_sigma: float = 2.0
    disparity_sigma: float = 2.0
    has_depth: bool = True


@dataclass
class Ellipsoid:
    """Constrained dual quadric: pose of ellipsoid frame in world + radii."""

    R: np.ndarray  # 3x3
    t: np.ndarray  # (3,)
    radii: np.ndarray  # (3,) positive

    def T(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def dual_matrix(self) -> np.ndarray:
        Q0 = np.diag(
            [self.radii[0] ** 2, self.radii[1] ** 2, self.radii[2] ** 2, -1.0]
        )
        Z = self.T()
        return Z @ Q0 @ Z.T

    def as_dict(self) -> dict:
        return {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "radii": self.radii.tolist(),
        }

    @staticmethod
    def from_vec(vec: np.ndarray) -> "Ellipsoid":
        from .geometry import rotvec_to_R

        R = rotvec_to_R(vec[0:3])
        t = vec[3:6].copy()
        radii = np.exp(vec[6:9])
        return Ellipsoid(R=R, t=t, radii=radii)

    def to_vec(self) -> np.ndarray:
        from .geometry import R_to_rotvec

        return np.concatenate([R_to_rotvec(self.R), self.t, np.log(np.maximum(self.radii, 1e-6))])


@dataclass
class SolverEstimate:
    ellipsoid: Ellipsoid
    # Each pose is T_world_from_left_camera (OpenCV convention)
    poses: list[np.ndarray] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)
    n_observations: int = 0
    rms: float = 0.0
    initialized: bool = False

    def pose_in_object_frame(self, T_wc: np.ndarray) -> np.ndarray:
        T_wo = self.ellipsoid.T()
        return np.linalg.inv(T_wo) @ T_wc

"""Modular dual-quadric pose/ellipsoid solver.

This package is intentionally free of web/UI dependencies so it can be
ported to Eigen/C++ for the LS1046-Space later. All math lives in
plain NumPy + SciPy least_squares.
"""

from .types import (
    CameraIntrinsics,
    StereoRig,
    AABB,
    OBB,
    ConvexHull,
    StereoObservation,
    Ellipsoid,
    SolverEstimate,
)
from .optimizer import QuadricSolver

__all__ = [
    "CameraIntrinsics",
    "StereoRig",
    "AABB",
    "OBB",
    "ConvexHull",
    "StereoObservation",
    "Ellipsoid",
    "SolverEstimate",
    "QuadricSolver",
]

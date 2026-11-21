"""Tests for boxrad.frames."""

from __future__ import annotations

import json

import numpy as np
import pytest

from boxrad.frames import Pose, default_x_axis, orthonormal_basis

RTOL = 1e-12
SEED = 42


def test_random_poses_orthonormal_and_roundtrip():
    rng = np.random.default_rng(SEED)
    for _ in range(1000):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        x_axis = rng.normal(size=3)
        if np.linalg.norm(x_axis - np.dot(x_axis, normal) * normal) < 1e-6:
            x_axis = np.array([0.0, 1.0, 0.0])
        pose = Pose.make(rng.uniform(-10, 10, size=3), normal, x_axis)
        R = pose.R
        assert np.allclose(R.T @ R, np.eye(3), atol=RTOL)
        assert R.shape == (3, 3)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=RTOL)
        z_hat = R[:, 2]
        n_hat = normal / np.linalg.norm(normal)
        assert np.linalg.norm(np.cross(z_hat, n_hat)) == pytest.approx(0.0, abs=RTOL)
        x_hat = R[:, 0]
        a = np.asarray(x_axis, dtype=np.float64)
        assert float(np.dot(x_hat, a)) > 0.0
        assert float(np.dot(x_hat, z_hat)) == pytest.approx(0.0, abs=RTOL)
        p = rng.uniform(-10, 10, size=3)
        err = np.linalg.norm(pose.to_world(pose.to_local(p)) - p)
        assert err <= RTOL


def test_known_basis_plus_y_normal():
    R = orthonormal_basis([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert np.allclose(R[:, 0], [1.0, 0.0, 0.0], atol=RTOL)
    assert np.allclose(R[:, 1], [0.0, 0.0, -1.0], atol=RTOL)
    assert np.allclose(R[:, 2], [0.0, 1.0, 0.0], atol=RTOL)


def test_identity_pose():
    pose = Pose.identity()
    assert np.allclose(pose.R, np.eye(3), atol=RTOL)


def test_orthonormal_errors():
    with pytest.raises(ValueError, match="non-zero"):
        orthonormal_basis([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="parallel"):
        orthonormal_basis([0.0, 0.0, 1.0], [0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        orthonormal_basis([0.0, 0.0, 1.0], [np.nan, 0.0, 0.0])


def test_default_x_axis_threshold():
    n = np.array([0.95, 0.31, 0.0], dtype=np.float64)
    n /= np.linalg.norm(n)
    ax = default_x_axis(n)
    assert np.allclose(ax, [0.0, 1.0, 0.0])
    ax_z = default_x_axis([0.0, 0.0, 1.0])
    assert np.allclose(ax_z, [1.0, 0.0, 0.0])


def test_T_to_world_batch_and_dict_roundtrip():
    pose = Pose.make([0.1, -0.2, 0.3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    via_T = (pose.T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    assert np.allclose(via_T, pose.to_world(pts), atol=RTOL)
    blob = json.loads(json.dumps(pose.to_dict()))
    pose2 = Pose.from_dict(blob)
    assert np.allclose(pose2.position, pose.position)
    assert np.allclose(pose2.R, pose.R, atol=RTOL)
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def example_directory(monkeypatch):
    """Import the source examples without downloading reconstruction data."""
    directory = Path(__file__).resolve().parents[3] / "examples" / "python"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.syspath_prepend(str(directory / "reconstruction_system"))
    verbosity = o3d.utility.get_verbosity_level()
    yield directory
    o3d.utility.set_verbosity_level(verbosity)


def registration_case(case):
    """Evaluate known transforms independently of stochastic feature matching."""
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                       [1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    transformation = np.eye(4)
    if case in ("translation", "failed_translation"):
        transformation[:3, 3] = [2., -3., 4.]
    elif case == "tiny_translation":
        transformation[:3, 3] = [1e-10, -2e-10, 3e-10]
    elif case in ("rotation", "failed_rotation"):
        transformation[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    target = o3d.geometry.PointCloud(source).transform(transformation)
    if case.startswith("failed"):
        target.translate([20., 20., 20.])
    elif case == "low_overlap":
        target_points = np.asarray(target.points).copy()
        target_points[1:] += 20.
        target.points = o3d.utility.Vector3dVector(target_points)
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.1, transformation)
    return source, target, result


@pytest.mark.parametrize("method", ("fgr", "ransac"))
@pytest.mark.parametrize("case, expected_success", (
    ("identity", True),
    ("translation", True),
    ("tiny_translation", True),
    ("rotation", True),
    ("failed_identity", False),
    ("failed_translation", False),
    ("failed_rotation", False),
    ("low_overlap", False),
))
def test_fragment_registration_success(example_directory, monkeypatch, method,
                                       case, expected_success):
    """Success follows correspondences and overlap, not the transform trace."""
    register_fragments = importlib.import_module("register_fragments")
    source, target, result = registration_case(case)
    function = "registration_" + method + "_based_on_feature_matching"
    monkeypatch.setattr(o3d.pipelines.registration, function,
                        lambda *args, **kwargs: result)
    config = {"voxel_size": 0.05, "global_registration": method}

    success, transformation, information = (
        register_fragments.register_point_cloud_fpfh(source, target, None, None,
                                                     config))

    assert success == expected_success
    if expected_success:
        np.testing.assert_allclose(transformation, result.transformation)
        assert information[5, 5] == len(source.points)
    else:
        np.testing.assert_array_equal(transformation, np.eye(4))
        np.testing.assert_array_equal(information, np.zeros((6, 6)))


@pytest.mark.parametrize("method", ("fgr", "ransac"))
@pytest.mark.parametrize(
    "case", ("identity", "translation", "tiny_translation", "rotation",
             "failed_identity", "failed_translation", "failed_rotation"))
def test_registration_benchmark_alignment(example_directory, monkeypatch,
                                          method, case):
    """Keep valid RANSAC alignments and the FGR benchmark's existing output."""
    source, target, result = registration_case(case)
    monkeypatch.setattr(o3d.data, "LivingRoomPointClouds",
                        lambda: SimpleNamespace(paths=["source", "target"]))
    monkeypatch.setattr(
        o3d.io, "read_point_cloud",
        lambda path: o3d.geometry.PointCloud(source
                                             if path == "source" else target))
    function = "registration_" + method + "_based_on_feature_matching"
    monkeypatch.setattr(o3d.pipelines.registration, function,
                        lambda *args, **kwargs: result)

    namespace = runpy.run_path(str(example_directory / "benchmark" /
                                   ("benchmark_" + method + ".py")),
                               run_name="__main__")

    expected_count = int(method == "fgr" or result.fitness > 0.)
    assert len(namespace["alignment"]) == expected_count
    if method == "ransac" and expected_count:
        np.testing.assert_allclose(namespace["alignment"][0][3],
                                   np.linalg.inv(result.transformation))

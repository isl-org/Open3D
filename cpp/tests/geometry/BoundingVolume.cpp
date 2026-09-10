// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/BoundingVolume.h"

#include <Eigen/Geometry>

#include "tests/Tests.h"

using namespace open3d::geometry;

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace open3d {
namespace tests {

namespace {

/// Rotation built from the 30, 45 and 60 degree Euler angles.
Matrix3d TestRotation() {
    return (Eigen::AngleAxisd(M_PI / 3.0, Vector3d::UnitZ()) *
            Eigen::AngleAxisd(M_PI / 4.0, Vector3d::UnitY()) *
            Eigen::AngleAxisd(M_PI / 6.0, Vector3d::UnitX()))
            .toRotationMatrix();
}

Matrix4d MakeTransform(const Matrix3d& linear, const Vector3d& translation) {
    Matrix4d transformation = Matrix4d::Identity();
    transformation.topLeftCorner<3, 3>() = linear;
    transformation.topRightCorner<3, 1>() = translation;
    return transformation;
}

OrientedBoundingBox TestBox() {
    return OrientedBoundingBox({1, 2, 3}, TestRotation(), {2, 4, 6});
}

std::vector<Vector3d> TransformPoints(const std::vector<Vector3d>& points,
                                      const Matrix4d& transformation) {
    std::vector<Vector3d> transformed;
    transformed.reserve(points.size());
    for (const Vector3d& point : points) {
        transformed.push_back(
                (transformation * point.homogeneous()).hnormalized());
    }
    return transformed;
}

}  // namespace

TEST(OrientedBoundingBox, TransformIdentity) {
    OrientedBoundingBox box = TestBox();
    box.Transform(Matrix4d::Identity());

    ExpectEQ(box.center_, Vector3d{1, 2, 3});
    ExpectEQ(box.R_, TestRotation());
    ExpectEQ(box.extent_, Vector3d{2, 4, 6});
}

// The corners of the transformed box must be the transformed corners, which is
// what transforming a point cloud with the same matrix produces.
TEST(OrientedBoundingBox, TransformRigid) {
    const Matrix4d transformation = MakeTransform(TestRotation(), {-4, 5, 0.5});
    const std::vector<Vector3d> expected_points =
            TransformPoints(TestBox().GetBoxPoints(), transformation);

    OrientedBoundingBox box = TestBox();
    box.Transform(transformation);

    ExpectEQ(box.GetBoxPoints(), expected_points);
    ExpectEQ(box.center_,
             Vector3d((transformation * Vector3d(1, 2, 3).homogeneous())
                              .hnormalized()));
    ExpectEQ(box.extent_, Vector3d{2, 4, 6});
    EXPECT_NEAR(box.R_.determinant(), 1.0, 1e-6);
}

TEST(OrientedBoundingBox, TransformUniformScale) {
    const Matrix4d transformation =
            MakeTransform(2.0 * TestRotation(), {-4, 5, 0.5});
    const std::vector<Vector3d> expected_points =
            TransformPoints(TestBox().GetBoxPoints(), transformation);

    OrientedBoundingBox box = TestBox();
    box.Transform(transformation);

    ExpectEQ(box.GetBoxPoints(), expected_points);
    ExpectEQ(box.extent_, Vector3d{4, 8, 12});
    EXPECT_NEAR(box.R_.determinant(), 1.0, 1e-6);
}

// Transforming twice matches transforming once with the composed matrix.
TEST(OrientedBoundingBox, TransformComposes) {
    const Matrix4d first = MakeTransform(2.0 * TestRotation(), {-4, 5, 0.5});
    const Matrix4d second = MakeTransform(
            Eigen::AngleAxisd(M_PI / 5.0, Vector3d::UnitX()).toRotationMatrix(),
            {1, -1, 2});

    OrientedBoundingBox composed = TestBox();
    composed.Transform(second * first);

    OrientedBoundingBox chained = TestBox();
    chained.Transform(first).Transform(second);

    ExpectEQ(chained.center_, composed.center_);
    ExpectEQ(chained.R_, composed.R_);
    ExpectEQ(chained.extent_, composed.extent_);
}

TEST(OrientedBoundingBox, TransformNonUniformScale) {
    OrientedBoundingBox box = TestBox();
    EXPECT_THROW(box.Transform(
                         MakeTransform(Matrix3d(Vector3d(1, 2, 3).asDiagonal()),
                                       Vector3d::Zero())),
                 std::runtime_error);
}

TEST(OrientedBoundingBox, TransformShear) {
    Matrix3d shear = Matrix3d::Identity();
    shear(0, 1) = 0.5;

    OrientedBoundingBox box = TestBox();
    EXPECT_THROW(box.Transform(MakeTransform(shear, Vector3d::Zero())),
                 std::runtime_error);
}

TEST(OrientedBoundingBox, TransformMirror) {
    OrientedBoundingBox box = TestBox();
    EXPECT_THROW(box.Transform(MakeTransform(-Matrix3d::Identity(),
                                             Vector3d::Zero())),
                 std::runtime_error);
}

TEST(OrientedBoundingBox, TransformProjective) {
    Matrix4d transformation = MakeTransform(TestRotation(), {1, 2, 3});
    transformation(3, 0) = 0.1;

    OrientedBoundingBox box = TestBox();
    EXPECT_THROW(box.Transform(transformation), std::runtime_error);
}

TEST(OrientedBoundingEllipsoid, TransformUniformScale) {
    const Matrix4d transformation =
            MakeTransform(2.0 * TestRotation(), {-4, 5, 0.5});

    OrientedBoundingEllipsoid ellipsoid({1, 2, 3}, Matrix3d::Identity(),
                                        {1, 2, 3});
    ellipsoid.Transform(transformation);

    ExpectEQ(ellipsoid.center_,
             Vector3d((transformation * Vector3d(1, 2, 3).homogeneous())
                              .hnormalized()));
    ExpectEQ(ellipsoid.R_, TestRotation());
    ExpectEQ(ellipsoid.radii_, Vector3d{2, 4, 6});
}

TEST(OrientedBoundingEllipsoid, TransformNonUniformScale) {
    OrientedBoundingEllipsoid ellipsoid({1, 2, 3}, Matrix3d::Identity(),
                                        {1, 2, 3});
    EXPECT_THROW(ellipsoid.Transform(
                         MakeTransform(Matrix3d(Vector3d(1, 2, 3).asDiagonal()),
                                       Vector3d::Zero())),
                 std::runtime_error);
}

}  // namespace tests
}  // namespace open3d

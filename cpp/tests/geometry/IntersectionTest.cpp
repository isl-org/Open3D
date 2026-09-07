// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/IntersectionTest.h"

#include <array>
#include <cmath>

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(IntersectionTest, TriangleTriangle3dIssue5117) {
    const std::array<Eigen::Vector3d, 3> p{
            Eigen::Vector3d(0.0, 0.13918686, 1.0),
            Eigen::Vector3d(0.0, 0.0, 1.1270161),
            Eigen::Vector3d(1.0, 0.0, 1.0284119)};
    const std::array<Eigen::Vector3d, 3> q{
            Eigen::Vector3d(1.0, 1.1269569, 0.0),
            Eigen::Vector3d(1.0, 0.03113556, 1.0),
            Eigen::Vector3d(2.0, 1.0189056, 0.0)};

    // These separated triangles used to be reported as intersecting until a
    // rigid rotation moved their plane values beyond an absolute tolerance.
    const auto expect_separated = [](const auto& p_vertices,
                                     const auto& q_vertices) {
        for (int p_offset = 0; p_offset < 3; ++p_offset) {
            for (int q_offset = 0; q_offset < 3; ++q_offset) {
                EXPECT_FALSE(geometry::IntersectionTest::TriangleTriangle3d(
                        p_vertices[p_offset], p_vertices[(p_offset + 1) % 3],
                        p_vertices[(p_offset + 2) % 3], q_vertices[q_offset],
                        q_vertices[(q_offset + 1) % 3],
                        q_vertices[(q_offset + 2) % 3]));
                EXPECT_FALSE(geometry::IntersectionTest::TriangleTriangle3d(
                        q_vertices[q_offset], q_vertices[(q_offset + 1) % 3],
                        q_vertices[(q_offset + 2) % 3], p_vertices[p_offset],
                        p_vertices[(p_offset + 1) % 3],
                        p_vertices[(p_offset + 2) % 3]));
            }
        }
    };
    expect_separated(p, q);

    const double sqrt_half = std::sqrt(0.5);
    Eigen::Matrix3d rotation;
    rotation << 1.0, 0.0, 0.0, 0.0, sqrt_half, sqrt_half, 0.0, -sqrt_half,
            sqrt_half;
    for (double scale : {1e-3, 1.0, 1e3}) {
        std::array<Eigen::Vector3d, 3> transformed_p;
        std::array<Eigen::Vector3d, 3> transformed_q;
        const Eigen::Vector3d translation(10.0, -20.0, 30.0);
        for (int i = 0; i < 3; ++i) {
            transformed_p[i] = scale * rotation * p[i] + translation;
            transformed_q[i] = scale * rotation * q[i] + translation;
        }
        expect_separated(transformed_p, transformed_q);
    }
}

TEST(IntersectionTest, TriangleTriangle3dNearParallelIntersection) {
    // Keep uncertain near-parallel cases in the legacy predicate: these two
    // triangles cross even though their normals are almost parallel.
    const Eigen::Vector3d p0(-1.0, -1.0, -2.0);
    const Eigen::Vector3d p1(1.0, -1.0, 0.0);
    const Eigen::Vector3d p2(0.0, 1.0, 1.0);
    const Eigen::Vector3d q0(-1.0, -1.0, -2.000002);
    const Eigen::Vector3d q1(1.0, -1.0, 0.000002);
    const Eigen::Vector3d q2(0.0, 1.0, 1.0);

    EXPECT_TRUE(geometry::IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0,
                                                               q1, q2));
}

TEST(IntersectionTest, TriangleTriangle3dNearDegenerateIntersection) {
    const Eigen::Vector3d p0(0.0, 0.0, 0.0);
    const Eigen::Vector3d p1(1.0, 1.0, 1.0);
    const Eigen::Vector3d p2(2.0, 2.0 + 1e-11, 2.0 - 1e-11);

    // Q is contained in P, but both triangles are nearly degenerate.
    const Eigen::Vector3d q0 = 0.25 * p1;
    const Eigen::Vector3d q1 = 0.5 * p1;
    const Eigen::Vector3d q2 = 0.25 * p2;

    EXPECT_TRUE(geometry::IntersectionTest::TriangleTriangle3d(p0, p1, p2, q0,
                                                               q1, q2));
    EXPECT_TRUE(geometry::IntersectionTest::TriangleTriangle3d(q0, q1, q2, p0,
                                                               p1, p2));
}

TEST(IntersectionTest, PointsCoplanar) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d p2(0, 1, 0);
    Eigen::Vector3d p3(1, 1, 0);

    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p1, p2, p3));
    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p0, p2, p3));
    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p1, p2, p2));
}

TEST(IntersectionTest, LinesMinimumDistance) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d q0(0, 1, 0);
    Eigen::Vector3d q1(1, 1, 0);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p0, q0, q1),
              -1.);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, q0, q0),
              -2.);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, q0, q1),
              -3.);

    Eigen::Vector3d u0(1, 0, 0);
    Eigen::Vector3d u1(1, 1, 0);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, u0, u1),
              0.);
}

TEST(IntersectionTest, LineSegmentsMinimumDistance) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d q0(0, 0, 1);
    Eigen::Vector3d q1(1, 0, 1);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);

    p0 = Eigen::Vector3d(0, 0, 0);
    p1 = Eigen::Vector3d(1, 0, 0);
    q0 = Eigen::Vector3d(2, 0, 0);
    q1 = Eigen::Vector3d(4, 0, 0);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);

    p0 = Eigen::Vector3d(0, 0, 0);
    p1 = Eigen::Vector3d(1, 0, 0);
    q0 = Eigen::Vector3d(0, 1, 0);
    q1 = Eigen::Vector3d(1, 4, 0);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);
}

}  // namespace tests
}  // namespace open3d

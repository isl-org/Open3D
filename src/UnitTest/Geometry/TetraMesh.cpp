// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Geometry/TetraMesh.h"
#include <Eigen/Geometry>
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, Constructor) {
    geometry::TetraMesh tm;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::TetraMesh,
              tm.GetGeometryType());
    EXPECT_EQ(3, tm.Dimension());

    // public member variables
    EXPECT_EQ(0u, tm.vertices_.size());
    EXPECT_EQ(0u, tm.tetras_.size());

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3d, tm.GetMinBound());
    ExpectEQ(Zero3d, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasTetras());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, DISABLED_MemberData) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, Clear) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector4i imin(0, 0, 0, 0);
    Vector4i imax(size - 1, size - 1, size - 1, size - 1);

    geometry::TetraMesh tm;

    tm.vertices_.resize(size);
    tm.tetras_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.tetras_, imin, imax, 0);

    EXPECT_FALSE(tm.IsEmpty());

    ExpectEQ(Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
    ExpectEQ(Vector3d(996.078431, 996.078431, 996.078431), tm.GetMaxBound());

    EXPECT_TRUE(tm.HasVertices());
    EXPECT_TRUE(tm.HasTetras());

    tm.Clear();

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3d, tm.GetMinBound());
    ExpectEQ(Zero3d, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasTetras());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, IsEmpty) {
    int size = 100;

    geometry::TetraMesh tm;

    EXPECT_TRUE(tm.IsEmpty());

    tm.vertices_.resize(size);

    EXPECT_FALSE(tm.IsEmpty());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, GetMinBound) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TetraMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, GetMaxBound) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TetraMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Vector3d(996.078431, 996.078431, 996.078431), tm.GetMaxBound());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, Transform) {
    vector<Vector3d> ref_vertices = {
            {1.411252, 4.274168, 3.130918}, {1.231757, 4.154505, 3.183678},
            {1.403168, 4.268779, 2.121679}, {1.456767, 4.304511, 2.640845},
            {1.620902, 4.413935, 1.851255}, {1.374684, 4.249790, 3.062485},
            {1.328160, 4.218773, 1.795728}, {1.713446, 4.475631, 1.860145},
            {1.409239, 4.272826, 2.011462}, {1.480169, 4.320113, 1.177780}};

    int size = 10;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TetraMesh tm;

    tm.vertices_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);

    Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;

    tm.Transform(transformation);

    ExpectEQ(ref_vertices, tm.vertices_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, OperatorAppend) {
    size_t size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector4i imin(0, 0, 0, 0);
    Vector4i imax(size - 1, size - 1, size - 1, size - 1);

    geometry::TetraMesh tm0;
    geometry::TetraMesh tm1;

    tm0.vertices_.resize(size);
    tm0.tetras_.resize(size);

    tm1.vertices_.resize(size);
    tm1.tetras_.resize(size);

    Rand(tm0.vertices_, dmin, dmax, 0);
    Rand(tm0.tetras_, imin, imax, 0);

    Rand(tm1.vertices_, dmin, dmax, 0);
    Rand(tm1.tetras_, imin, imax, 0);

    geometry::TetraMesh tm(tm0);
    tm += tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertices_[i], tm.vertices_[i + 0]);
        ExpectEQ(tm1.vertices_[i], tm.vertices_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.tetras_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.tetras_[i], tm.tetras_[i + 0]);
        ExpectEQ(Vector4i(tm1.tetras_[i](0, 0) + size,
                          tm1.tetras_[i](1, 0) + size,
                          tm1.tetras_[i](2, 0) + size,
                          tm1.tetras_[i](3, 0) + size),
                 tm.tetras_[i + size]);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, OperatorADD) {
    size_t size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector4i imin(0, 0, 0, 0);
    Vector4i imax(size - 1, size - 1, size - 1, size - 1);

    geometry::TetraMesh tm0;
    geometry::TetraMesh tm1;

    tm0.vertices_.resize(size);
    tm0.tetras_.resize(size);

    tm1.vertices_.resize(size);
    tm1.tetras_.resize(size);

    Rand(tm0.vertices_, dmin, dmax, 0);
    Rand(tm0.tetras_, imin, imax, 0);

    Rand(tm1.vertices_, dmin, dmax, 0);
    Rand(tm1.tetras_, imin, imax, 0);

    geometry::TetraMesh tm = tm0 + tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertices_[i], tm.vertices_[i + 0]);
        ExpectEQ(tm1.vertices_[i], tm.vertices_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.tetras_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.tetras_[i], tm.tetras_[i + 0]);
        ExpectEQ(Vector4i(tm1.tetras_[i](0, 0) + size,
                          tm1.tetras_[i](1, 0) + size,
                          tm1.tetras_[i](2, 0) + size,
                          tm1.tetras_[i](3, 0) + size),
                 tm.tetras_[i + size]);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, Purge) {
    typedef vector<Vector4i, aligned_allocator<Vector4i>> vector_Vector4i;

    vector<Vector3d> vertices = {// duplicate
                                 {796.078431, 909.803922, 196.078431},
                                 {796.078431, 909.803922, 196.078431},
                                 {333.333333, 764.705882, 274.509804},
                                 {552.941176, 474.509804, 627.450980},
                                 {364.705882, 509.803922, 949.019608},
                                 {913.725490, 635.294118, 713.725490},
                                 {141.176471, 603.921569, 15.686275},
                                 // unreferenced
                                 {239.215686, 133.333333, 803.921569}};

    vector_Vector4i tetras = {{2, 6, 3, 4},
                              // same tetra after vertex duplicate is removed
                              {0, 2, 3, 4},
                              {1, 2, 3, 4},
                              // duplicates
                              {3, 4, 5, 6},
                              {4, 3, 5, 6},
                              {3, 4, 6, 5},
                              {5, 3, 4, 6},
                              // degenerate
                              {4, 5, 6, 6}};

    geometry::TetraMesh tm;
    tm.vertices_ = vertices;
    tm.tetras_ = tetras;

    tm.RemoveDuplicatedTetras();

    vector_Vector4i ref_tetras_after_tetra_duplicate_removal = {
            {2, 6, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4}, {3, 4, 5, 6}, {4, 5, 6, 6}

    };
    ExpectEQ(ref_tetras_after_tetra_duplicate_removal, tm.tetras_);

    tm.RemoveDuplicatedVertices();

    vector<Vector3d> ref_vertices_after_duplicate_removal = {
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275},
            {239.215686, 133.333333, 803.921569}};
    vector_Vector4i ref_tetras_after_vertex_duplicate_removal = {{1, 5, 2, 3},
                                                                 {0, 1, 2, 3},
                                                                 {0, 1, 2, 3},
                                                                 {2, 3, 4, 5},
                                                                 {3, 4, 5, 5}};
    ExpectEQ(ref_vertices_after_duplicate_removal, tm.vertices_);
    ExpectEQ(ref_tetras_after_vertex_duplicate_removal, tm.tetras_);

    tm.RemoveUnreferencedVertices();

    vector<Vector3d> ref_vertices_after_unreferenced_removal = {
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275}};
    ExpectEQ(ref_vertices_after_unreferenced_removal, tm.vertices_);

    tm.RemoveDegenerateTetras();

    vector_Vector4i ref_tetras_after_degenerate_removal = {
            {1, 5, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {2, 3, 4, 5}

    };
    ExpectEQ(ref_tetras_after_degenerate_removal, tm.tetras_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, HasVertices) {
    int size = 100;

    geometry::TetraMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    tm.vertices_.resize(size);

    EXPECT_TRUE(tm.HasVertices());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, HasTetras) {
    int size = 100;

    geometry::TetraMesh tm;

    EXPECT_FALSE(tm.HasTetras());

    tm.vertices_.resize(size);
    tm.tetras_.resize(size);

    EXPECT_TRUE(tm.HasTetras());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, CreateFromPointCloud) {
    // create a random point cloud
    geometry::PointCloud pc;

    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);

    Rand(pc.points_, dmin, dmax, 0);

    auto tm = geometry::TetraMesh::CreateFromPointCloud(pc);

    EXPECT_EQ(pc.points_.size(), tm->vertices_.size());

    // check if the delaunay property holds for all tetras.
    // There should be no vertex inside the circumsphere of a tetrahedron.
    // This is more a check to see if we have used the right Qhull parameters.
    auto tetrahedron_circumsphere = [&](const geometry::TetraMesh& tm,
                                        size_t tetra_idx) {
        Vector4i tetra = tm.tetras_[tetra_idx];
        Matrix4d homogeneous_points;
        for (int i = 0; i < 4; ++i) {
            homogeneous_points.row(i) = tm.vertices_[tetra[i]].homogeneous();
        }
        double det = homogeneous_points.determinant();
        Vector4d b;
        for (int i = 0; i < 4; ++i) {
            b[i] = -tm.vertices_[tetra[i]].squaredNorm();
        }

        Matrix4d tmp;
        double x[3];
        for (int i = 0; i < 3; ++i) {
            tmp = homogeneous_points;
            tmp.col(i) = b;
            x[i] = tmp.determinant() / det;
        }
        Vector3d center(-x[0] / 2, -x[1] / 2, -x[2] / 2);
        double sqr_radius = (tm.vertices_[tetra[0]] - center).squaredNorm();

        return std::make_tuple(center, sqr_radius);
    };

    bool circumsphere_property = true;
    for (size_t i = 0; i < tm->tetras_.size() && circumsphere_property; ++i) {
        Vector3d sphere_center;
        double sqr_radius;
        std::tie(sphere_center, sqr_radius) = tetrahedron_circumsphere(*tm, i);

        const Vector4i tetra = tm->tetras_[i];

        for (int j = 0; j < (int)tm->vertices_.size(); ++j) {
            if (tetra[0] == j || tetra[1] == j || tetra[2] == j ||
                tetra[3] == j) {
                continue;
            }
            Vector3d v = tm->vertices_[j];
            double sqr_dist_center_v = (v - sphere_center).squaredNorm();
            if (sqr_dist_center_v <= sqr_radius) {
                circumsphere_property = false;
                break;
            }
        }
    }
    EXPECT_TRUE(circumsphere_property);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TetraMesh, ExtractTriangleMesh) {
    // create a random point cloud
    geometry::PointCloud pc;

    int size = 250;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1.0, 1.0, 1.0);

    pc.points_.resize(size);

    Rand(pc.points_, dmin, dmax, 0);

    auto tm = geometry::TetraMesh::CreateFromPointCloud(pc);

    vector<double> values(tm->vertices_.size());

    // distance to center as values
    Vector3d center(0.5 * (dmin + dmax));
    for (size_t i = 0; i < tm->vertices_.size(); ++i) {
        double dist = (tm->vertices_[i] - center).norm();
        values[i] = dist;
    }

    // all values are below the level
    {
        auto triangle_mesh = tm->ExtractTriangleMesh(values, 123.4);
        EXPECT_EQ(triangle_mesh->vertices_.size(), size_t(0));
        EXPECT_EQ(triangle_mesh->triangles_.size(), size_t(0));
    }

    // all values are above the level
    {
        auto triangle_mesh = tm->ExtractTriangleMesh(values, -123.4);
        EXPECT_EQ(triangle_mesh->vertices_.size(), size_t(0));
        EXPECT_EQ(triangle_mesh->triangles_.size(), size_t(0));
    }

    // values below and above the level
    {
        auto triangle_mesh = tm->ExtractTriangleMesh(values, 0.25);
        EXPECT_EQ(triangle_mesh->vertices_.size(), size_t(125));
        EXPECT_EQ(triangle_mesh->triangles_.size(), size_t(246));

        // the following should have no effect
        triangle_mesh->RemoveDuplicatedVertices();
        triangle_mesh->RemoveDuplicatedTriangles();
        triangle_mesh->RemoveUnreferencedVertices();
        triangle_mesh->RemoveDegenerateTriangles();

        EXPECT_EQ(triangle_mesh->vertices_.size(), size_t(125));
        EXPECT_EQ(triangle_mesh->triangles_.size(), size_t(246));
    }
}

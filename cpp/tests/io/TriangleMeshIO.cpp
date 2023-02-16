// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/io/TriangleMeshIO.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshIO, DISABLED_CreateMeshFromFile) { NotImplemented(); }

TEST(TriangleMeshIO, DISABLED_ReadTriangleMesh) { NotImplemented(); }

TEST(TriangleMeshIO, DISABLED_WriteTriangleMesh) { NotImplemented(); }

TEST(TriangleMeshIO, DISABLED_ReadTriangleMeshFromPLY) { NotImplemented(); }

TEST(TriangleMeshIO, DISABLED_WriteTriangleMeshToPLY) { NotImplemented(); }

TEST(TriangleMeshIO, AddTrianglesByEarClippingNonconvexPoly) {
    // This test checks if a bug in AddTrianglesByEarClipping() is fixed.

    auto poly = std::make_shared<geometry::TriangleMesh>();
    poly->vertices_.resize(5);

    //  1      3
    //  |\    /|
    //  | \2 / |    y
    //  |  \/  |    |
    //  |______|    /-- x
    //  0      4   z
    Eigen::Vector3d v1(0., 0., 0.);
    Eigen::Vector3d v2(0., 2., 0.);
    Eigen::Vector3d v3(1., 1., 0.);
    Eigen::Vector3d v4(2., 2., 0.);
    Eigen::Vector3d v5(2., 0., 0.);
    poly->vertices_[0] = v1;
    poly->vertices_[1] = v2;
    poly->vertices_[2] = v3;
    poly->vertices_[3] = v4;
    poly->vertices_[4] = v5;
    std::vector<unsigned int> indices{0, 1, 2, 3, 4};

    EXPECT_TRUE(io::AddTrianglesByEarClipping(*poly, indices));
}

}  // namespace tests
}  // namespace open3d

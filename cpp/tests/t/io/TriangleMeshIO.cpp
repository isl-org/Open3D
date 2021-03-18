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

#include "tests/UnitTest.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/io/TriangleMeshIO.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshIO, ReadTriangleMesh) { 
    t::geometry::TriangleMesh mesh;
    EXPECT_TRUE(t::io::ReadTriangleMesh(TEST_DATA_DIR "/knot.ply", mesh));
    EXPECT_EQ( mesh.GetTriangles().GetLength(), 2880);
    EXPECT_EQ( mesh.GetVertices().GetLength(), 1440);
}

TEST(TriangleMeshIO, WriteTriangleMesh) { 
    t::geometry::TriangleMesh mesh1, mesh2;
    EXPECT_TRUE(t::io::ReadTriangleMesh(TEST_DATA_DIR "/knot.ply", mesh1));
    EXPECT_TRUE(t::io::WriteTriangleMesh("test.ply", mesh1));
    EXPECT_TRUE(t::io::ReadTriangleMesh("test.ply", mesh2));
    EXPECT_EQ( mesh1.GetTriangles().GetLength(), mesh2.GetTriangles().GetLength());
    EXPECT_EQ( mesh1.GetVertices().GetLength(), mesh2.GetVertices().GetLength());
 }

}  // namespace tests
}  // namespace open3d

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

#include "Open3D/Geometry/TexturedTriangleMesh.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/IntersectionTest.h"
#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/Qhull.h"

#include <Eigen/Dense>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

TexturedTriangleMesh &TexturedTriangleMesh::Clear() {
    TriangleMesh::Clear();
    uvs_.clear();
    texture_.Clear();
    return *this;
}

TexturedTriangleMesh &TexturedTriangleMesh::operator+=(
        const TexturedTriangleMesh &mesh) {
    // TODO: need to copy image into a new one and update uv coordinates
    throw std::runtime_error("Not implemented");
}

TexturedTriangleMesh TexturedTriangleMesh::operator+(
        const TexturedTriangleMesh &mesh) const {
    return (TexturedTriangleMesh(*this) += mesh);
}

}  // namespace geometry
}  // namespace open3d

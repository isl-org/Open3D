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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace geometry {

class Image;

class TexturedTriangleMesh : public TriangleMesh {
public:
    TexturedTriangleMesh()
        : TriangleMesh(Geometry::GeometryType::TexturedTriangleMesh) {}
    ~TexturedTriangleMesh() override {}

public:
    virtual TexturedTriangleMesh &Clear() override;

public:
    TexturedTriangleMesh &operator+=(const TexturedTriangleMesh &mesh);
    TexturedTriangleMesh operator+(const TexturedTriangleMesh &mesh) const;

    // assumes for each triangle we have three uv coordinates
    bool HasUvs() const {
        return HasTriangles() && uvs_.size() == 3 * triangles_.size();
    }

    bool HasTexture() const { return !texture_.IsEmpty(); }

protected:
    // Forward child class type to avoid indirect nonvirtual base
    TexturedTriangleMesh(Geometry::GeometryType type) : TriangleMesh(type) {}

public:
    // 3 uv coordinates per triangle
    std::vector<Eigen::Vector2d> uvs_;
    Image texture_;
};

}  // namespace geometry
}  // namespace open3d

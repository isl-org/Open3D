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

#pragma once

#include "open3d/visualization/rendering/Material.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class DrawableGeometry
///
/// \brief Mix-in class for geometry types that can be visualized
class DrawableGeometry {
public:
    DrawableGeometry() {}
    ~DrawableGeometry() {}

    /// Check if a material has been applied to this Geometry with SetMaterial.
    bool HasMaterial() const { return material_.IsValid(); }

    /// Get material associated with this Geometry.
    visualization::rendering::Material &GetMaterial() { return material_; }

    /// Get const reference to material associated with this Geometry
    const visualization::rendering::Material &GetMaterial() const {
        return material_;
    }

    /// Set the material properties associate with this Geometry
    void SetMaterial(const visualization::rendering::Material &material) {
        material_ = material;
    }

private:
    /// Material associated with this geometry
    visualization::rendering::Material material_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d

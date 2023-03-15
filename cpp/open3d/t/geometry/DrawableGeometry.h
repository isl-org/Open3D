// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

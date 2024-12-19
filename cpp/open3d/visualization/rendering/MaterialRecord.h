// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/geometry/Image.h"
#include "open3d/visualization/rendering/Gradient.h"
#include "open3d/visualization/utility/GLHelper.h"

namespace open3d {
namespace visualization {
namespace rendering {

struct MaterialRecord {
    std::string name;

    // Rendering attributes
    bool has_alpha = false;

    // PBR Material properties and maps
    Eigen::Vector4f base_color = Eigen::Vector4f(1.f, 1.f, 1.f, 1.f);
    float base_metallic = 0.f;
    float base_roughness = 1.f;
    float base_reflectance = 0.5f;
    float base_clearcoat = 0.f;
    float base_clearcoat_roughness = 0.f;
    float base_anisotropy = 0.f;
    Eigen::Vector4f emissive_color = Eigen::Vector4f(0.f, 0.f, 0.f, 1.f);

    // PBR material properties for refractive materials
    float thickness = 1.f;
    float transmission = 1.f;
    Eigen::Vector3f absorption_color =
            Eigen::Vector3f(1.f, 1.f, 1.f);  // linear color
    float absorption_distance = 1.f;

    float point_size = 3.f;
    float line_width = 1.f;  // only used with shader = "unlitLine"

    std::shared_ptr<geometry::Image> albedo_img;
    std::shared_ptr<geometry::Image> normal_img;
    std::shared_ptr<geometry::Image> ao_img;
    std::shared_ptr<geometry::Image> metallic_img;
    std::shared_ptr<geometry::Image> roughness_img;
    std::shared_ptr<geometry::Image> reflectance_img;
    std::shared_ptr<geometry::Image> clearcoat_img;
    std::shared_ptr<geometry::Image> clearcoat_roughness_img;
    std::shared_ptr<geometry::Image> anisotropy_img;

    // Combined images
    std::shared_ptr<geometry::Image> ao_rough_metal_img;

    // Colormap (incompatible with other settings except point_size)
    // Values for 'value' must be in [0, 1] and the vector must be sorted
    // by increasing value. 'shader' must be "unlitGradient".
    std::shared_ptr<Gradient> gradient;
    float scalar_min = 0.0f;
    float scalar_max = 1.0f;

    // Colors are assumed to be sRGB and tone-mapped accordingly.
    // If tone-mapping is disabled, then colors would be in linear RGB space,
    // in which case this should be set to false. If necessary, colors will be
    // linearized on the CPU.
    bool sRGB_color = false;

    // Unlike the material property sRGB_color which is used to indicate that
    // source colors are in sRGB colorspace, sRGB_vertex_color indicates that
    // per-vertex colors are in sRGB space and should be passed to the GPU as
    // sRGB color.
    bool sRGB_vertex_color = false;

    // Background image (shader = "unlitBackground")
    float aspect_ratio = 0.0f;  // 0: uses base_color; >0: uses albedo_img

    // Infinite ground plane
    float ground_plane_axis = 0.f;  // 0: XZ; >0: XY; <0: YZ

    // Generic material properties
    std::unordered_map<std::string, Eigen::Vector4f> generic_params;
    std::unordered_map<std::string, geometry::Image> generic_imgs;

    std::string shader = "defaultUnlit";
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

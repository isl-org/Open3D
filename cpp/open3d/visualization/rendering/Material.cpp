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

#include "open3d/visualization/rendering/Material.h"

#include "open3d/visualization/rendering/MaterialRecord.h"

namespace open3d {
namespace visualization {
namespace rendering {

void Material::SetDefaultProperties() {
    material_name_ = "defaultUnlit";
    SetBaseColor(Eigen::Vector4f(1.f, 1.f, 1.f, 1.f));
    SetBaseMetallic(0.f);
    SetBaseRoughness(1.f);
    SetBaseReflectance(0.5f);
    SetBaseClearcoat(0.f);
    SetBaseClearcoatRoughness(0.f);
    SetAnisotropy(0.f);
    SetThickness(1.f);
    SetTransmission(1.f);
    SetAbsorptionColor(Eigen::Vector4f(1.f, 1.f, 1.f, 1.f));
    SetAbsorptionDistance(1.f);
    SetPointSize(3.f);
    SetLineWidth(1.f);
}

void Material::SetTextureMap(const std::string &key,
                             const t::geometry::Image &image) {
    // The Image must be on the CPU since GPU-resident images are not currently
    // supported. Also, we copy the Image here because the image data is
    // asynchronously copied to the GPU and we want to make sure the Image data
    // doesn't get modified while being copied.
    texture_maps_[key] = image.To(core::Device("CPU:0"), true);
}

void Material::ToMaterialRecord(MaterialRecord &record) const {
    record.shader = GetMaterialName();
    // Convert base material properties
    if (HasBaseColor()) {
        record.base_color = GetBaseColor();
    }
    if (HasBaseMetallic()) {
        record.base_metallic = GetBaseMetallic();
    }
    if (HasBaseRoughness()) {
        record.base_roughness = GetBaseRoughness();
    }
    if (HasBaseReflectance()) {
        record.base_reflectance = GetBaseReflectance();
    }
    if (HasBaseClearcoat()) {
        record.base_clearcoat = GetBaseClearcoat();
    }
    if (HasBaseClearcoatRoughness()) {
        record.base_clearcoat_roughness = GetBaseClearcoatRoughness();
    }
    if (HasAnisotropy()) {
        record.base_anisotropy = GetAnisotropy();
    }
    if (HasThickness()) {
        record.thickness = GetThickness();
    }
    if (HasTransmission()) {
        record.transmission = GetTransmission();
    }
    if (HasAbsorptionColor()) {
        record.absorption_color = Eigen::Vector3f(GetAbsorptionColor().data());
    }
    if (HasAbsorptionDistance()) {
        record.absorption_distance = GetAbsorptionDistance();
    }
    // Convert maps
    if (HasAlbedoMap()) {
        record.albedo_img =
                std::make_shared<geometry::Image>(GetAlbedoMap().ToLegacy());
    }
    if (HasNormalMap()) {
        record.normal_img =
                std::make_shared<geometry::Image>(GetNormalMap().ToLegacy());
    }
    if (HasAOMap()) {
        record.ao_img =
                std::make_shared<geometry::Image>(GetAOMap().ToLegacy());
    }
    if (HasMetallicMap()) {
        record.metallic_img =
                std::make_shared<geometry::Image>(GetMetallicMap().ToLegacy());
    }
    if (HasRoughnessMap()) {
        record.roughness_img =
                std::make_shared<geometry::Image>(GetRoughnessMap().ToLegacy());
    }
    if (HasReflectanceMap()) {
        record.reflectance_img = std::make_shared<geometry::Image>(
                GetReflectanceMap().ToLegacy());
    }
    if (HasClearcoatMap()) {
        record.clearcoat_img =
                std::make_shared<geometry::Image>(GetClearcoatMap().ToLegacy());
    }
    if (HasClearcoatRoughnessMap()) {
        record.clearcoat_roughness_img = std::make_shared<geometry::Image>(
                GetClearcoatRoughnessMap().ToLegacy());
    }
    if (HasAnisotropyMap()) {
        record.anisotropy_img = std::make_shared<geometry::Image>(
                GetAnisotropyMap().ToLegacy());
    }
    if (HasAORoughnessMetalMap()) {
        record.ao_rough_metal_img = std::make_shared<geometry::Image>(
                GetAORoughnessMetalMap().ToLegacy());
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

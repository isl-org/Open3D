// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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
    SetEmissiveColor(Eigen::Vector4f(0.f, 0.f, 0.f, 1.f));
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

std::string Material::ToString() const {
    if (!IsValid()) {
        return "Invalid Material\n";
    }
    std::ostringstream os;
    os << "Material " << material_name_ << '\n';
    for (const auto &kv : scalar_properties_) {
        os << '\t' << kv.first << ": " << kv.second << '\n';
    }
    for (const auto &kv : vector_properties_) {
        os << '\t' << kv.first << ": " << kv.second.transpose() << '\n';
    }
    for (const auto &kv : texture_maps_) {
        os << '\t' << kv.first << ": " << kv.second.ToString() << '\n';
    }
    return os.str();
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
    if (HasEmissiveColor()) {
        record.emissive_color = GetEmissiveColor();
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
    if (HasPointSize()) {
        record.point_size = GetPointSize();
    }
    if (HasLineWidth()) {
        record.line_width = GetLineWidth();
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

Material Material::FromMaterialRecord(const MaterialRecord &record) {
    using t::geometry::Image;
    Material tmat(record.shader);
    // scalar and vector properties
    tmat.SetBaseColor(record.base_color);
    tmat.SetBaseMetallic(record.base_metallic);
    tmat.SetBaseRoughness(record.base_roughness);
    tmat.SetBaseReflectance(record.base_reflectance);
    tmat.SetBaseClearcoat(record.base_clearcoat);
    tmat.SetBaseClearcoatRoughness(record.base_clearcoat_roughness);
    tmat.SetAnisotropy(record.base_anisotropy);
    tmat.SetEmissiveColor(record.emissive_color);
    // refractive materials
    tmat.SetThickness(record.thickness);
    tmat.SetTransmission(record.transmission);
    tmat.SetAbsorptionDistance(record.absorption_distance);
    // points and lines
    tmat.SetPointSize(record.point_size);
    tmat.SetLineWidth(record.line_width);
    // maps
    if (record.albedo_img) {
        tmat.SetAlbedoMap(Image::FromLegacy(*record.albedo_img));
    }
    if (record.normal_img) {
        tmat.SetNormalMap(Image::FromLegacy(*record.normal_img));
    }
    if (record.ao_img) {
        tmat.SetAOMap(Image::FromLegacy(*record.ao_img));
    }
    if (record.metallic_img) {
        tmat.SetMetallicMap(Image::FromLegacy(*record.metallic_img));
    }
    if (record.roughness_img) {
        tmat.SetRoughnessMap(Image::FromLegacy(*record.roughness_img));
    }
    if (record.reflectance_img) {
        tmat.SetReflectanceMap(Image::FromLegacy(*record.reflectance_img));
    }
    if (record.clearcoat_img) {
        tmat.SetClearcoatMap(Image::FromLegacy(*record.clearcoat_img));
    }
    if (record.clearcoat_roughness_img) {
        tmat.SetClearcoatRoughnessMap(
                Image::FromLegacy(*record.clearcoat_roughness_img));
    }
    if (record.anisotropy_img) {
        tmat.SetAnisotropyMap(Image::FromLegacy(*record.anisotropy_img));
    }
    if (record.ao_rough_metal_img) {
        tmat.SetAORoughnessMetalMap(
                Image::FromLegacy(*record.ao_rough_metal_img));
    }

    return tmat;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

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

#include <string>

#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace visualization {
namespace rendering {

struct MaterialRecord;

class Material {
public:
    using TextureMaps = std::unordered_map<std::string, t::geometry::Image>;
    using ScalarPropertyMap = std::unordered_map<std::string, float>;
    using VectorPropertyMap = std::unordered_map<
            std::string,
            Eigen::Vector4f,
            std::hash<std::string>,
            std::equal_to<std::string>,
            Eigen::aligned_allocator<
                    std::pair<const std::string, Eigen::Vector4f>>>;

    /// Create an empty, invalid material
    Material() = default;

    Material(const Material &mat) = default;

    /// Create an empty but valid material for the specified material name
    Material(const std::string &material_name)
        : material_name_(material_name) {}

    /// Sets a number of properties to the defaults expected by Open3D
    /// shaders
    void SetDefaultProperties();

    /// Returns true if the Material was not created with the default
    /// constructor and therefore has a valid shader name.
    bool IsValid() const { return !material_name_.empty(); }

    /// Get the name of the material.
    const std::string &GetMaterialName() const { return material_name_; }

    /// Returns the texture map map
    const TextureMaps &GetTextureMaps() const { return texture_maps_; }

    /// Get images (texture maps) of this Material. Throws exception if the
    /// image does not exist.
    ///
    /// \param key Map name
    const t::geometry::Image &GetTextureMap(const std::string &key) const {
        return texture_maps_.at(key);
    }

    /// Returns the map of scalar properties
    const ScalarPropertyMap &GetScalarProperties() const {
        return scalar_properties_;
    }

    /// Get scalar properties of this Material. Throws exception if the property
    /// does not exist.
    ///
    /// \param key Property name
    float GetScalarProperty(const std::string &key) const {
        return scalar_properties_.at(key);
    }

    /// Returns the map of vector properties
    const VectorPropertyMap &GetVectorProperties() const {
        return vector_properties_;
    }

    /// Get vector properties of this Material. Throws exception if the property
    /// does not exist.
    ///
    /// \param key Property name
    Eigen::Vector4f GetVectorProperty(const std::string &key) const {
        return vector_properties_.at(key);
    }

    /// Set texture map. If map already exists it is overwritten, otherwise a
    /// new key/image will be created.
    ///
    /// \param key map name
    /// \param image Image associated with map name
    void SetTextureMap(const std::string &key, const t::geometry::Image &image);

    /// Set scalar property. If property already exists it is overwritten,
    /// otherwise a new key/value will be created.
    ///
    /// \param key property name
    /// \param value Value to assign to property name
    void SetScalarProperty(const std::string &key, float value) {
        scalar_properties_[key] = value;
    }

    /// Set vector property. If property already exists it is overwritten,
    /// otherwise a new key/value will be created.
    ///
    /// \param key property name
    /// \param value Value to assign to property name
    void SetVectorProperty(const std::string &key,
                           const Eigen::Vector4f &value) {
        vector_properties_[key] = value;
    }

    /// Set material name. The material name should match the name of a built
    // in or user specified shader. The name is NOT checked to ensure it is
    // valid.
    ///
    /// \param shader The name of the shader.
    void SetMaterialName(const std::string &material_name) {
        material_name_ = material_name;
    }

    /// Return true if the map exists
    ///
    /// \param key Map name
    bool HasTextureMap(const std::string &key) const {
        return texture_maps_.count(key) > 0;
    }

    /// Return true if the property exists
    ///
    /// \param key Property name
    bool HasScalarProperty(const std::string &key) const {
        return scalar_properties_.count(key) > 0;
    }

    /// Return true if the property exists
    ///
    /// \param key Property name
    bool HasVectorProperty(const std::string &key) const {
        return vector_properties_.count(key) > 0;
    }

    ////////////////////////////////////////////////////////////////////////////
    ///
    /// The following are convenience methods for common PBR material properties
    ///
    ////////////////////////////////////////////////////////////////////////////
    const t::geometry::Image &GetAlbedoMap() const {
        return GetTextureMap("albedo");
    }
    const t::geometry::Image &GetNormalMap() const {
        return GetTextureMap("normal");
    }
    const t::geometry::Image &GetAOMap() const {
        return GetTextureMap("ambient_occlusion");
    }
    const t::geometry::Image &GetMetallicMap() const {
        return GetTextureMap("metallic");
    }
    const t::geometry::Image &GetRoughnessMap() const {
        return GetTextureMap("roughness");
    }
    const t::geometry::Image &GetReflectanceMap() const {
        return GetTextureMap("reflectance");
    }
    const t::geometry::Image &GetClearcoatMap() const {
        return GetTextureMap("clear_coat");
    }
    const t::geometry::Image &GetClearcoatRoughnessMap() const {
        return GetTextureMap("clear_coat_roughness");
    }
    const t::geometry::Image &GetAnisotropyMap() const {
        return GetTextureMap("anisotropy");
    }
    /// Ambient occlusion, roughness, and metallic maps in a single 3 channel
    /// texture. Commonly used in glTF models.
    const t::geometry::Image &GetAORoughnessMetalMap() const {
        return GetTextureMap("ao_rough_metal");
    }

    bool HasAlbedoMap() const { return HasTextureMap("albedo"); }
    bool HasNormalMap() const { return HasTextureMap("normal"); }
    bool HasAOMap() const { return HasTextureMap("ambient_occlusion"); }
    bool HasMetallicMap() const { return HasTextureMap("metallic"); }
    bool HasRoughnessMap() const { return HasTextureMap("roughness"); }
    bool HasReflectanceMap() const { return HasTextureMap("reflectance"); }
    bool HasClearcoatMap() const { return HasTextureMap("clear_coat"); }
    bool HasClearcoatRoughnessMap() const {
        return HasTextureMap("clear_coat_roughness");
    }
    bool HasAnisotropyMap() const { return HasTextureMap("anisotropy"); }
    bool HasAORoughnessMetalMap() const {
        return HasTextureMap("ao_rough_metal");
    }

    void SetAlbedoMap(const t::geometry::Image &image) {
        SetTextureMap("albedo", image);
    }
    void SetNormalMap(const t::geometry::Image &image) {
        SetTextureMap("normal", image);
    }
    void SetAOMap(const t::geometry::Image &image) {
        SetTextureMap("ambient_occlusion", image);
    }
    void SetMetallicMap(const t::geometry::Image &image) {
        SetTextureMap("metallic", image);
    }
    void SetRoughnessMap(const t::geometry::Image &image) {
        SetTextureMap("roughness", image);
    }
    void SetReflectanceMap(const t::geometry::Image &image) {
        SetTextureMap("reflectance", image);
    }
    void SetClearcoatMap(const t::geometry::Image &image) {
        SetTextureMap("clear_coat", image);
    }
    void SetClearcoatRoughnessMap(const t::geometry::Image &image) {
        SetTextureMap("clear_coat_roughness", image);
    }
    void SetAnisotropyMap(const t::geometry::Image &image) {
        SetTextureMap("anisotropy", image);
    }
    void SetAORoughnessMetalMap(const t::geometry::Image &image) {
        SetTextureMap("ao_rough_metal", image);
    }

    Eigen::Vector4f GetBaseColor() const {
        return GetVectorProperty("base_color");
    }
    float GetBaseMetallic() const { return GetScalarProperty("metallic"); }
    float GetBaseRoughness() const { return GetScalarProperty("roughness"); }
    float GetBaseReflectance() const {
        return GetScalarProperty("reflectance");
    }
    float GetBaseClearcoat() const { return GetScalarProperty("clear_coat"); }
    float GetBaseClearcoatRoughness() const {
        return GetScalarProperty("clear_coat_roughness");
    }
    float GetAnisotropy() const { return GetScalarProperty("anisotropy"); }
    float GetThickness() const { return GetScalarProperty("thickness"); }
    float GetTransmission() const { return GetScalarProperty("transmission"); }
    Eigen::Vector4f GetAbsorptionColor() const {
        return GetVectorProperty("absorption_color");
    }
    float GetAbsorptionDistance() const {
        return GetScalarProperty("absorption_distance");
    }

    bool HasBaseColor() const { return HasVectorProperty("color"); }
    bool HasBaseMetallic() const { return HasScalarProperty("metallic"); }
    bool HasBaseRoughness() const { return HasScalarProperty("roughness"); }
    bool HasBaseReflectance() const { return HasScalarProperty("reflectance"); }
    bool HasBaseClearcoat() const { return HasScalarProperty("clear_coat"); }
    bool HasBaseClearcoatRoughness() const {
        return HasScalarProperty("clear_coat_roughness");
    }
    bool HasAnisotropy() const { return HasScalarProperty("anisotropy"); }
    bool HasThickness() const { return HasScalarProperty("thickness"); }
    bool HasTransmission() const { return HasScalarProperty("transmission"); }
    bool HasAbsorptionColor() const {
        return HasVectorProperty("absorption_color");
    }
    bool HasAbsorptionDistance() const {
        return HasScalarProperty("absorption_distance");
    }

    void SetBaseColor(const Eigen::Vector4f &value) {
        SetVectorProperty("base_color", value);
    }
    void SetBaseMetallic(float value) { SetScalarProperty("metallic", value); }
    void SetBaseRoughness(float value) {
        SetScalarProperty("roughness", value);
    }
    void SetBaseReflectance(float value) {
        SetScalarProperty("reflectance", value);
    }
    void SetBaseClearcoat(float value) {
        SetScalarProperty("clear_coat", value);
    }
    void SetBaseClearcoatRoughness(float value) {
        SetScalarProperty("clear_coat_roughness", value);
    }
    void SetAnisotropy(float value) { SetScalarProperty("anisotropy", value); }
    void SetThickness(float value) { SetScalarProperty("thickness", value); }
    void SetTransmission(float value) {
        SetScalarProperty("transmission", value);
    }
    void SetAbsorptionColor(const Eigen::Vector4f &value) {
        SetVectorProperty("absorption_color", value);
    }
    void SetAbsorptionDistance(float value) {
        SetScalarProperty("absorption_distance", value);
    }

    ////////////////////////////////////////////////////////////////////////////
    ///
    /// The following are convenience methods for Pointclouds and LineSet
    /// shaders
    ///
    ////////////////////////////////////////////////////////////////////////////
    float GetPointSize() const { return GetScalarProperty("point_size"); }
    float GetLineWidth() const { return GetScalarProperty("line_width"); }

    bool HasPointSize() const { return HasScalarProperty("point_size"); }
    bool HasLineWidth() const { return HasScalarProperty("line_width"); }

    void SetPointSize(float value) { SetScalarProperty("point_size", value); }
    void SetLineWidth(float value) { SetScalarProperty("line_width", value); }

    /// Fills a legacy MaterialRecord constructed from this Material
    void ToMaterialRecord(MaterialRecord &record) const;

private:
    std::string material_name_;
    TextureMaps texture_maps_;
    ScalarPropertyMap scalar_properties_;
    VectorPropertyMap vector_properties_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

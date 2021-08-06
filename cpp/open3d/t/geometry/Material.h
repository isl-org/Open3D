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
namespace t {
namespace geometry {

class Material {
public:
    /// Create an empty, invalid material
    Material() = default;

    Material(const Material &mat) = default;

    /// Create an empty but valid material for the specified shader name
    Material(const std::string &shader) : shader_(shader) {}

    /// Sets a number of properties to the defaults expected by Open3D
    /// shaders
    void SetDefaultProperties();

    /// Returns true if the Material was not created with the default
    /// constructor and therefore has a valid shader name.
    bool IsValid() const { return shader_ != "INVALID"; }

    /// Get the name of the shader for this material
    const std::string &GetShaderName() const { return shader_; }

    /// Get images (texture maps) of this Material. Throws exception if the
    /// image does not exist.
    ///
    /// \param key Map name
    Image &GetTextureMap(const std::string &key) {
        return texture_maps_.at(key);
    }

    /// Get scalar properties of this Material. Throws exception if the property
    /// does not exist.
    ///
    /// \param key Property name
    float GetScalarProperty(const std::string &key) {
        return scalar_properties_.at(key);
    }

    /// Get vector properties of this Material. Throws exception if the property
    /// does not exist.
    ///
    /// \param key Property name
    Eigen::Vector4f GetVectorProperty(const std::string &key) {
        return vector_properties_.at(key);
    }

    /// Set texture map. If map already exists it is overwritten, otherwise a
    /// new key/image will be created.
    ///
    /// \param key map name
    /// \param image Image associated with map name
    void SetTextureMap(const std::string &key, const Image &image);

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

    /// Set shader name. The shader name should match the name of a built in or
    /// user specified shader. The name is NOT checked to esnure it is valid.
    ///
    /// \param shader The name of the shader.
    void SetShaderName(const std::string &shader) { shader_ = shader; }

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
    Image &GetAlbedoMap() { return GetTextureMap("albedo"); }
    Image &GetNormalMap() { return GetTextureMap("normal"); }
    Image &GetAOMap() { return GetTextureMap("ao"); }
    Image &GetMetallicMap() { return GetTextureMap("metallic"); }
    Image &GetRoughnessMap() { return GetTextureMap("roughness"); }
    Image &GetReflectanceMap() { return GetTextureMap("reflectance"); }
    Image &GetClearcoatMap() { return GetTextureMap("clearcoat"); }
    Image &GetClearcoatRoughnessMap() {
        return GetTextureMap("clearcoat_roughness");
    }
    Image &GetAnisotropyMap() { return GetTextureMap("anisotropy"); }
    Image &GetAORoughnessMetalMap() { return GetTextureMap("ao_rough_metal"); }

    bool HasAlbedoMap() { return HasTextureMap("albedo"); }
    bool HasNormalMap() { return HasTextureMap("normal"); }
    bool HasAOMap() { return HasTextureMap("ao"); }
    bool HasMetallicMap() { return HasTextureMap("metallic"); }
    bool HasRoughnessMap() { return HasTextureMap("roughness"); }
    bool HasReflectanceMap() { return HasTextureMap("reflectance"); }
    bool HasClearcoatMap() { return HasTextureMap("clearcoat"); }
    bool HasClearcoatRoughnessMap() {
        return HasTextureMap("clearcoat_roughness");
    }
    bool HasAnisotropyMap() { return HasTextureMap("anisotropy"); }
    bool HasAORoughnessMetalMap() { return HasTextureMap("ao_rough_metal"); }

    void SetAlbedoMap(const Image &image) { SetTextureMap("albedo", image); }
    void SetNormalMap(const Image &image) { SetTextureMap("normal", image); }
    void SetAOMap(const Image &image) { SetTextureMap("ao", image); }
    void SetMetallicMap(const Image &image) {
        SetTextureMap("metallic", image);
    }
    void SetRoughnessMap(const Image &image) {
        SetTextureMap("roughness", image);
    }
    void SetReflectanceMap(const Image &image) {
        SetTextureMap("reflectance", image);
    }
    void SetClearcoatMap(const Image &image) {
        SetTextureMap("clearcoat", image);
    }
    void SetClearcoatRoughnessMap(const Image &image) {
        SetTextureMap("clearcoat_roughness", image);
    }
    void SetAnisotropyMap(const Image &image) {
        SetTextureMap("anisotropy", image);
    }
    void SetAORoughnessMetalMap(const Image &image) {
        SetTextureMap("ao_rough_metal", image);
    }

    Eigen::Vector4f GetBaseColor() { return GetVectorProperty("base_color"); }
    float GetBaseMetallic() { return GetScalarProperty("base_metallic"); }
    float GetBaseRoughness() { return GetScalarProperty("base_roughness"); }
    float GetBaseReflectance() { return GetScalarProperty("base_reflectance"); }
    float GetBaseClearcoat() { return GetScalarProperty("base_clearcoat"); }
    float GetBaseClearcoatRoughness() {
        return GetScalarProperty("base_clearcoat_roughness");
    }
    float GetAnisotropy() { return GetScalarProperty("base_anisotropy"); }
    float GetThickness() { return GetScalarProperty("thickness"); }
    float GetTransmission() { return GetScalarProperty("transmission"); }
    Eigen::Vector4f GetAbsorptionColor() {
        return GetVectorProperty("absorption_color");
    }
    float GetAbsorptionDistance() {
        return GetScalarProperty("absorption_distance");
    }

    bool HasBaseColor() { return HasVectorProperty("base_color"); }
    bool HasBaseMetallic() { return HasScalarProperty("base_metallic"); }
    bool HasBaseRoughness() { return HasScalarProperty("base_roughness"); }
    bool HasBaseReflectance() { return HasScalarProperty("base_reflectance"); }
    bool HasBaseClearcoat() { return HasScalarProperty("base_clearcoat"); }
    bool HasBaseClearcoatRoughness() {
        return HasScalarProperty("base_clearcoat_roughness");
    }
    bool HasAnisotropy() { return HasScalarProperty("base_anisotropy"); }
    bool HasThickness() { return HasScalarProperty("thickness"); }
    bool HasTransmission() { return HasScalarProperty("transmission"); }
    bool HasAbsorptionColor() { return HasVectorProperty("absorption_color"); }
    bool HasAbsorptionDistance() {
        return HasScalarProperty("absorption_distance");
    }

    void SetBaseColor(const Eigen::Vector4f &value) {
        SetVectorProperty("base_color", value);
    }
    void SetBaseMetallic(float value) {
        SetScalarProperty("base_metallic", value);
    }
    void SetBaseRoughness(float value) {
        SetScalarProperty("base_roughness", value);
    }
    void SetBaseReflectance(float value) {
        SetScalarProperty("base_reflectance", value);
    }
    void SetBaseClearcoat(float value) {
        SetScalarProperty("base_clearcoat", value);
    }
    void SetBaseClearcoatRoughness(float value) {
        SetScalarProperty("base_clearcoat_roughness", value);
    }
    void SetAnisotropy(float value) {
        SetScalarProperty("base_anisotropy", value);
    }
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
    float GetPointSize() { return GetScalarProperty("point_size"); }
    float GetLineWidth() { return GetScalarProperty("line_width"); }

    bool HasPointSize() { return HasScalarProperty("point_size"); }
    bool HasLineWidth() { return HasScalarProperty("line_width"); }

    void SetPointSize(float value) { SetScalarProperty("point_size", value); }
    void SetLineWidth(float value) { SetScalarProperty("line_width", value); }

private:
    std::string shader_ = "INVALID";
    std::unordered_map<std::string, Image> texture_maps_;
    std::unordered_map<std::string, float> scalar_properties_;
    std::unordered_map<std::string, Eigen::Vector4f> vector_properties_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d

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
    /// Creates an invalid material
    Material();
    Material(const std::string &shader);

    bool IsValid() const { return shader_ != "INVALID"; }

    const std::string &GetShaderName() const { return shader_; }

    Image &GetTextureMap(const std::string &map_name) {
        return texture_maps_.at(map_name);
    }

    bool HasTextureMap(const std::string &map_name) const {
        return texture_maps_.count(map_name) > 0;
    }

    float GetScalarProperty(const std::string &property_name) {
        return scalar_properties_.at(property_name);
    }

    bool HasScalarProperty(const std::string &property_name) const {
        return scalar_properties_.count(property_name) > 0;
    }

    Eigen::Vector4f GetVectorProperty(const std::string &property_name) {
        return vector_properties_.at(property_name);
    }

    bool HasVectorProperty(const std::string &property_name) const {
        return vector_properties_.count(property_name) > 0;
    }

    // Convenience accessors similar to TriangleMesh?
    // Could have convenience methods for each of the properties in Material
    // struct
    Image &GetAlbedoMap() { return GetTextureMap("albedo"); }
    Image &GetNormalMap() { return GetTextureMap("normal"); }
    // etc....

private:
    std::string shader_;
    std::unordered_map<std::string, Image> texture_maps_;
    std::unordered_map<std::string, float> scalar_properties_;
    std::unordered_map<std::string, Eigen::Vector4f> vector_properties_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d

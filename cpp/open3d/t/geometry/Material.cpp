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

#include "open3d/t/geometry/Material.h"

namespace open3d {
namespace t {
namespace geometry {

void Material::SetDefaultProperties() {
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

void Material::SetTextureMap(const std::string &key, const Image &image) {
    // Image must be on CPU. If Image is already on CPU the following does not
    // perform an uneccesasry copy
    texture_maps_[key] = image.CPU();
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <memory>

#include "open3d/visualization/rendering/MaterialModifier.h"

/// @cond
namespace filament {
class MaterialInstance;
class TextureSampler;
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentMaterialModifier : public MaterialModifier {
public:
    static filament::TextureSampler SamplerFromSamplerParameters(
            const TextureSamplerParameters& sampler_config);

    FilamentMaterialModifier(const std::shared_ptr<filament::MaterialInstance>&
                                     material_instance,
                             const MaterialInstanceHandle& id);
    FilamentMaterialModifier() = default;

    void Reset();
    void
    Init(const std::shared_ptr<filament::MaterialInstance>& material_instance,
         const MaterialInstanceHandle& id);

    MaterialModifier& SetParameter(const char* parameter, int value) override;
    MaterialModifier& SetParameter(const char* parameter, float value) override;
    MaterialModifier& SetParameter(const char* parameter,
                                   const Eigen::Vector3f& value) override;
    MaterialModifier& SetColor(const char* parameter,
                               const Eigen::Vector3f& value,
                               bool srgb) override;
    MaterialModifier& SetColor(const char* parameter,
                               const Eigen::Vector4f& value,
                               bool srgb) override;

    MaterialModifier& SetTexture(
            const char* parameter,
            const TextureHandle& texture,
            const TextureSamplerParameters& sampler) override;

    MaterialModifier& SetDoubleSided(bool double_sided) override;

    MaterialInstanceHandle Finish() override;

private:
    MaterialInstanceHandle current_handle_;
    std::shared_ptr<filament::MaterialInstance> material_instance_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

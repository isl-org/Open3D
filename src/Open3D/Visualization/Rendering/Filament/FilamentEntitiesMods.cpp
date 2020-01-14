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

#include "FilamentEntitiesMods.h"

#include "FilamentEngine.h"
#include "FilamentResourceManager.h"

#include "Open3D/Utility/Console.h"

#include <filament/MaterialInstance.h>

namespace open3d {
namespace visualization {

namespace {

using namespace filament;

TextureSampler::WrapMode ConvertWrapMode(TextureSamplerParameters::WrapMode mode) {
    switch (mode) {
        case TextureSamplerParameters::WrapMode::ClampToEdge:
            return TextureSampler::WrapMode::CLAMP_TO_EDGE;
        case TextureSamplerParameters::WrapMode::Repeat:
            return TextureSampler::WrapMode::REPEAT;
        case TextureSamplerParameters::WrapMode::MirroredRepeat:
            return TextureSampler::WrapMode::MIRRORED_REPEAT;
    }

    return TextureSampler::WrapMode::CLAMP_TO_EDGE;
}

TextureSampler SamplerFromSamplerParameters(
        const TextureSamplerParameters& samplerConfig) {
    TextureSampler sampler;

    switch (samplerConfig.filterMag) {
        case TextureSamplerParameters::MagFilter::Nearest:
            sampler.setMagFilter(TextureSampler::MagFilter::NEAREST);
            break;
        case TextureSamplerParameters::MagFilter::Linear:
            sampler.setMagFilter(TextureSampler::MagFilter::LINEAR);
            break;
    }

    switch (samplerConfig.filterMin) {
        case TextureSamplerParameters::MinFilter::Nearest:
            sampler.setMinFilter(TextureSampler::MinFilter::NEAREST);
            break;
        case TextureSamplerParameters::MinFilter::Linear:
            sampler.setMinFilter(TextureSampler::MinFilter::LINEAR);
            break;
        case TextureSamplerParameters::MinFilter::NearestMipmapNearest:
            sampler.setMinFilter(
                    TextureSampler::MinFilter::NEAREST_MIPMAP_NEAREST);
            break;
        case TextureSamplerParameters::MinFilter::LinearMipmapNearest:
            sampler.setMinFilter(
                    TextureSampler::MinFilter::LINEAR_MIPMAP_NEAREST);
            break;
        case TextureSamplerParameters::MinFilter::NearestMipmapLinear:
            sampler.setMinFilter(
                    TextureSampler::MinFilter::NEAREST_MIPMAP_LINEAR);
            break;
        case TextureSamplerParameters::MinFilter::LinearMipmapLinear:
            sampler.setMinFilter(
                    TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR);
            break;
    }

    sampler.setWrapModeS(ConvertWrapMode(samplerConfig.wrapU));
    sampler.setWrapModeT(ConvertWrapMode(samplerConfig.wrapV));
    sampler.setWrapModeR(ConvertWrapMode(samplerConfig.wrapW));

    sampler.setAnisotropy(sampler.getAnisotropy());

    return sampler;
}
}

void FilamentMaterialModifier::Reset() {
    if (materialInstance_ != nullptr) {
        utility::LogWarning("Previous material instance modifications are not finished!");
    }

    materialInstance_ = nullptr;
    currentHandle_ = MaterialInstanceHandle::kBad;
}

void FilamentMaterialModifier::InitWithMaterialInstance(
        const std::shared_ptr<filament::MaterialInstance>& aMaterialInstance,
        const MaterialInstanceHandle& id) {
    if (materialInstance_ != nullptr) {
        utility::LogWarning("Previous material instance modifications are not finished!");
    }

    materialInstance_ = aMaterialInstance;
    currentHandle_ = id;
}

MaterialModifier& FilamentMaterialModifier::SetParameter(const char* parameter,
                                                         const float value) {
    if (materialInstance_) {
        materialInstance_->setParameter(parameter, value);
    }

    return *this;
}

MaterialModifier& FilamentMaterialModifier::SetColor(
        const char* parameter, const Eigen::Vector3f& value) {
    if (materialInstance_) {
        const auto color =
                filament::math::float3{value.x(), value.y(), value.z()};
        materialInstance_->setParameter(parameter, filament::RgbType::sRGB,
                                       color);
    }

    return *this;
}

MaterialModifier& FilamentMaterialModifier::SetTexture(
        const char* parameter,
        const TextureHandle& textureHandle,
        const TextureSamplerParameters& samplerConfig) {
    if (materialInstance_) {
        auto wTexture =
                EngineInstance::GetResourceManager().GetTexture(textureHandle);

        if (auto texturePtr = wTexture.lock()) {
            filament::TextureSampler sampler(TextureSampler::MinFilter::LINEAR,
                                             TextureSampler::MagFilter::LINEAR);

            materialInstance_->setParameter(
                    parameter, texturePtr.get(),
                    SamplerFromSamplerParameters(samplerConfig));
        } else {
            utility::LogWarning(
                    "Failed to set texture for material.\n\tMaterial handle: {}\n\tTexture handle: {}\n\tParameter name: {}",
                    currentHandle_, textureHandle, parameter);
        }
    }

    return *this;
}

MaterialInstanceHandle FilamentMaterialModifier::Finish() {
    auto res = currentHandle_;

    materialInstance_ = nullptr;
    currentHandle_ = MaterialInstanceHandle::kBad;

    return res;
}

}
}
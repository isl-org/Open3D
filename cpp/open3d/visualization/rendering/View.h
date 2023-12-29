// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>

#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace visualization {
namespace rendering {

class Scene;
class Camera;
class ColorGradingParams;

class View {
public:
    enum class TargetBuffers : std::uint8_t {
        None = 0u,
        Color = 1u,
        Depth = 2u,
        Stencil = 4u,

        ColorAndDepth = Color | Depth,
        ColorAndStencil = Color | Stencil,
        DepthAndStencil = Depth | Stencil,
        All = Color | Depth | Stencil
    };

    enum class Mode : std::uint8_t {
        Color = 0u,
        Depth,
        Normals,
        // This three modes always stay at end
        ColorMapX,
        ColorMapY,
        ColorMapZ
    };

    enum class ShadowType : std::uint8_t {
        kPCF, /* Percentage closer filter, the default */
        kVSM  /* Variance shadow map */
    };

    virtual ~View() {}

    virtual void SetDiscardBuffers(const TargetBuffers& buffers) = 0;
    virtual Mode GetMode() const = 0;
    virtual void SetMode(Mode mode) = 0;
    virtual void SetWireframe(bool enable) = 0;

    virtual void SetSampleCount(int n) = 0;
    virtual int GetSampleCount() const = 0;

    virtual void SetViewport(std::int32_t x,
                             std::int32_t y,
                             std::uint32_t w,
                             std::uint32_t h) = 0;
    virtual std::array<int, 4> GetViewport() const = 0;

    virtual void SetPostProcessing(bool enabled) = 0;
    virtual void SetAmbientOcclusion(bool enabled,
                                     bool ssct_enabled = false) = 0;
    virtual void SetBloom(bool enabled,
                          float strength = 0.5f,
                          int spread = 6) = 0;
    virtual void SetAntiAliasing(bool enabled, bool temporal = false) = 0;
    virtual void SetShadowing(bool enabled, ShadowType type) = 0;

    virtual void SetColorGrading(const ColorGradingParams& color_grading) = 0;

    virtual void ConfigureForColorPicking() = 0;

    virtual void EnableViewCaching(bool enable) = 0;
    virtual bool IsCached() const = 0;
    virtual TextureHandle GetColorBuffer() = 0;

    virtual Camera* GetCamera() const = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

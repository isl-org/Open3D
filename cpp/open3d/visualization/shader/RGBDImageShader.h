// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/Image.h"
#include "open3d/visualization/shader/ShaderWrapper.h"

namespace open3d {
namespace visualization {

namespace glsl {

enum ImageTextureMode { Depth = 0, RGB = 1, Grayscale = 2 };

class RGBDImageShader : public ShaderWrapper {
public:
    ~RGBDImageShader() override { Release(); }

protected:
    RGBDImageShader(const std::string &name) : ShaderWrapper(name) {
        Compile();
    }

protected:
    bool Compile() final;
    void Release() final;
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
    void UnbindGeometry() final;

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_UV_;
    GLuint vertex_UV_buffer_;
    GLuint image_texture_;
    GLuint color_texture_buffer_;
    GLuint depth_texture_;
    GLuint depth_texture_buffer_;
    GLuint vertex_scale_;
    GLuint texture_mode_;
    GLuint depth_max_;
    float depth_max_data_;
    float color_rel_ratio_ = 0.5f;

    /* Switches corresponding to the glsl shader */
    ImageTextureMode depth_texture_mode_;
    ImageTextureMode color_texture_mode_;
    gl_util::GLVector3f vertex_scale_data_;
};

class RGBDImageShaderForImage : public RGBDImageShader {
public:
    RGBDImageShaderForImage() : RGBDImageShader("RGBDImageShaderForImage") {}

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) final;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

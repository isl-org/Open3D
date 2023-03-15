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

class ImageMaskShader : public ShaderWrapper {
public:
    ~ImageMaskShader() override { Release(); }

protected:
    ImageMaskShader(const std::string &name) : ShaderWrapper(name) {
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
                                const ViewControl &view,
                                geometry::Image &image) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_UV_;
    GLuint vertex_UV_buffer_;
    GLuint image_texture_;
    GLuint image_texture_buffer_;
    GLuint mask_color_;
    GLuint mask_alpha_;

    gl_util::GLVector3f mask_color_data_;
    GLfloat mask_alpha_data_;
};

class ImageMaskShaderForImage : public ImageMaskShader {
public:
    ImageMaskShaderForImage() : ImageMaskShader("ImageMaskShaderForImage") {}

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) final;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                geometry::Image &render_image) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

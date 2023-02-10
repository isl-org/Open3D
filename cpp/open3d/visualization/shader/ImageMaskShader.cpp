// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/shader/ImageMaskShader.h"

#include <algorithm>

#include "open3d/geometry/Image.h"
#include "open3d/visualization/shader/Shader.h"
#include "open3d/visualization/visualizer/RenderOptionWithEditing.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool ImageMaskShader::Compile() {
    if (!CompileShaders(ImageMaskVertexShader, NULL, ImageMaskFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_UV_ = glGetAttribLocation(program_, "vertex_UV");
    image_texture_ = glGetUniformLocation(program_, "image_texture");
    mask_color_ = glGetUniformLocation(program_, "mask_color");
    mask_alpha_ = glGetUniformLocation(program_, "mask_alpha");
    return true;
}

void ImageMaskShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool ImageMaskShader::BindGeometry(const geometry::Geometry &geometry,
                                   const RenderOption &option,
                                   const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    geometry::Image render_image;
    if (!PrepareBinding(geometry, option, view, render_image)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    const GLfloat vertex_position_buffer_data[18] = {
            -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
    };
    const GLfloat vertex_UV_buffer_data[12] = {
            0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    };
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_position_buffer_data),
                 vertex_position_buffer_data, GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_UV_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_UV_buffer_data),
                 vertex_UV_buffer_data, GL_STATIC_DRAW);

    glGenTextures(1, &image_texture_buffer_);
    glBindTexture(GL_TEXTURE_2D, image_texture_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, render_image.width_,
                 render_image.height_, 0, GL_RED, GL_UNSIGNED_BYTE,
                 render_image.data_.data());

    if (option.interpolation_option_ ==
        RenderOption::TextureInterpolationOption::Nearest) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    bound_ = true;
    return true;
}

bool ImageMaskShader::RenderGeometry(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    glUseProgram(program_);
    glUniform3fv(mask_color_, 1, mask_color_data_.data());
    glUniform1fv(mask_alpha_, 1, &mask_alpha_data_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, image_texture_buffer_);
    glUniform1i(image_texture_, 0);
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_UV_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glVertexAttribPointer(vertex_UV_, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_UV_);

    return true;
}

void ImageMaskShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_UV_buffer_);
        glDeleteTextures(1, &image_texture_buffer_);
        bound_ = false;
    }
}

bool ImageMaskShaderForImage::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Image) {
        PrintShaderWarning("Rendering type is not geometry::Image.");
        return false;
    }
    const geometry::Image &image = (const geometry::Image &)geometry;
    if (image.width_ != view.GetWindowWidth() ||
        image.height_ != view.GetWindowHeight()) {
        PrintShaderWarning("Mask image does not match framebuffer size.");
        return false;
    }
    const auto &_option = (RenderOptionWithEditing &)option;
    mask_color_data_ = _option.selection_polygon_mask_color_.cast<float>();
    mask_alpha_data_ = (float)_option.selection_polygon_mask_alpha_;
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

bool ImageMaskShaderForImage::PrepareBinding(const geometry::Geometry &geometry,
                                             const RenderOption &option,
                                             const ViewControl &view,
                                             geometry::Image &render_image) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Image) {
        PrintShaderWarning("Rendering type is not geometry::Image.");
        return false;
    }
    const geometry::Image &image = (const geometry::Image &)geometry;
    if (!image.HasData()) {
        PrintShaderWarning("Binding failed with empty image.");
        return false;
    }
    if (image.width_ != view.GetWindowWidth() ||
        image.height_ != view.GetWindowHeight()) {
        PrintShaderWarning("Mask image does not match framebuffer size.");
        return false;
    }
    render_image.Prepare(image.width_, image.height_, 1, 1);
    for (int i = 0; i < image.height_ * image.width_; i++) {
        render_image.data_[i] = (image.data_[i] != 0) * 255;
    }
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = 6;
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

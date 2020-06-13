// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "RGBDImageShader.h"

#include <Open3D/Open3D.h>
#include <algorithm>

#include "Open3D/Geometry/Image.h"
#include "Open3D/Visualization/Shader/Shader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool RGBDImageShader::Compile() {
    if (CompileShaders(ImageVertexShader, NULL, RGBDImageFragmentShader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_UV_ = glGetAttribLocation(program_, "vertex_UV");
    image_texture_ = glGetUniformLocation(program_, "image_texture");
    vertex_scale_ = glGetUniformLocation(program_, "vertex_scale");

    /* Use an option to switch between modes */
    texture_mode_ = glGetUniformLocation(program_, "texture_mode");
    depth_max_ = glGetUniformLocation(program_, "depth_max");
    return true;
}

void RGBDImageShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool RGBDImageShader::BindGeometry(const geometry::Geometry &geometry,
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
    if (!PrepareBinding(geometry, option, view)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }
    const geometry::RGBDImage &rgbd = (const geometry::RGBDImage &)geometry;

    const geometry::Image &color_image = rgbd.color_;
    const geometry::Image &depth_image = rgbd.depth_;

    // Create buffers and bind the geometry
    const GLfloat vertex_position_buffer_data[18] = {
            -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
    };
    const GLfloat vertex_UV_buffer_data[12] = {
            0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_position_buffer_data),
                 vertex_position_buffer_data, GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_UV_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_UV_buffer_data),
                 vertex_UV_buffer_data, GL_STATIC_DRAW);

    GLuint color_fmt = color_image.num_of_channels_ == 1 ? GL_RED : GL_RGB;
    GLuint color_type =
            color_image.bytes_per_channel_ == 1 ? GL_UNSIGNED_BYTE : GL_FLOAT;
    color_texture_mode_ = (color_fmt == GL_RGB) ? ImageTextureMode::RGB
                                                : ImageTextureMode::Grayscale;
    glGenTextures(1, &color_texture_buffer_);
    glBindTexture(GL_TEXTURE_2D, color_texture_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, color_image.width_,
                 color_image.height_, 0, color_fmt, color_type,
                 color_image.data_.data());
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

    GLuint depth_fmt = GL_RED;
    GLuint depth_type =
            depth_image.bytes_per_channel_ == 2 ? GL_UNSIGNED_SHORT : GL_FLOAT;
    depth_texture_mode_ = ImageTextureMode::Depth;

    /* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
     * In OpenGL, texture of GL_UNSIGNED_SHORT are converted to [0, 1] (float),
     * while texture of GL_FLOAT seems not converted. */
    depth_max_data_ =
            (depth_type == GL_UNSIGNED_SHORT)
                    ? static_cast<float>(option.image_max_depth_) / 65535.0f
                    : static_cast<float>(option.image_max_depth_) / 1000.0f;
    glGenTextures(1, &depth_texture_buffer_);
    glBindTexture(GL_TEXTURE_2D, depth_texture_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, depth_image.width_,
                 depth_image.height_, 0, depth_fmt, depth_type,
                 depth_image.data_.data());
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

bool RGBDImageShader::RenderGeometry(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    glUseProgram(program_);
    glUniform3fv(vertex_scale_, 1, vertex_scale_data_.data());
    glUniform1f(depth_max_, depth_max_data_);
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_UV_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glVertexAttribPointer(vertex_UV_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(image_texture_, 0);

    glViewport(0, 0,
               static_cast<GLsizei>(view.GetWindowWidth() * color_rel_ratio_),
               view.GetWindowHeight());
    glUniform1i(texture_mode_, color_texture_mode_);
    glBindTexture(GL_TEXTURE_2D, color_texture_buffer_);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glViewport(
            static_cast<GLint>(view.GetWindowWidth() * color_rel_ratio_), 0,
            static_cast<GLint>(view.GetWindowWidth() * (1 - color_rel_ratio_)),
            view.GetWindowHeight());
    glUniform1i(texture_mode_, depth_texture_mode_);
    glBindTexture(GL_TEXTURE_2D, depth_texture_buffer_);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_UV_);

    return true;
}

void RGBDImageShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_UV_buffer_);
        glDeleteTextures(1, &color_texture_buffer_);
        glDeleteTextures(1, &depth_texture_buffer_);
        bound_ = false;
    }
}

bool RGBDImageShaderForImage::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::RGBDImage) {
        PrintShaderWarning("Rendering type is not geometry::RGBDImage.");
        return false;
    }
    const geometry::RGBDImage &rgbd = (const geometry::RGBDImage &)geometry;
    const geometry::Image &image = rgbd.color_;

    int color_width = rgbd.color_.width_;
    int color_height = rgbd.color_.height_;
    int depth_width = rgbd.depth_.width_;
    int depth_height = rgbd.depth_.height_;

    float depth_rel_width = color_height * ((float)depth_width / depth_height);
    color_rel_ratio_ = (float)color_width / (color_width + depth_rel_width);

    GLfloat ratio_x, ratio_y;
    switch (option.image_stretch_option_) {
        case RenderOption::ImageStretchOption::StretchKeepRatio:
            ratio_x = GLfloat(image.width_) /
                      GLfloat(view.GetWindowWidth() * color_rel_ratio_);
            ratio_y = GLfloat(image.height_) / GLfloat(view.GetWindowHeight());
            if (ratio_x < ratio_y) {
                ratio_x /= ratio_y;
                ratio_y = 1.0f;
            } else {
                ratio_y /= ratio_x;
                ratio_x = 1.0f;
            }
            break;
        case RenderOption::ImageStretchOption::StretchWithWindow:
            ratio_x = 1.0f;
            ratio_y = 1.0f;
            break;
        case RenderOption::ImageStretchOption::OriginalSize:
        default:
            ratio_x = GLfloat(image.width_) /
                      GLfloat(view.GetWindowWidth() * color_rel_ratio_);
            ratio_y = GLfloat(image.height_) / GLfloat(view.GetWindowHeight());
            break;
    }
    vertex_scale_data_(0) = ratio_x;
    vertex_scale_data_(1) = ratio_y;
    vertex_scale_data_(2) = 1.0f;
    glDisable(GL_DEPTH_TEST);
    return true;
}

bool RGBDImageShaderForImage::PrepareBinding(const geometry::Geometry &geometry,
                                             const RenderOption &option,
                                             const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::RGBDImage) {
        PrintShaderWarning("Rendering type is not geometry::RGBDImage.");
        return false;
    }
    const geometry::RGBDImage rgbd = (const geometry::RGBDImage &)geometry;
    if (rgbd.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty image.");
        return false;
    }

    bool is_color_rgb = (rgbd.color_.num_of_channels_ == 3 &&
                         rgbd.color_.bytes_per_channel_ == 1);
    bool is_color_grayscale = (rgbd.color_.num_of_channels_ == 1 &&
                               rgbd.color_.bytes_per_channel_ == 4);
    bool is_depth_ushort = (rgbd.depth_.num_of_channels_ == 1 &&
                            rgbd.depth_.bytes_per_channel_ == 2);
    bool is_depth_float = (rgbd.depth_.num_of_channels_ == 1 &&
                           rgbd.depth_.bytes_per_channel_ == 4);
    if (!is_color_rgb && !is_color_grayscale) {
        PrintShaderWarning("Binding failed with incorrect color image format.");
        return false;
    }
    if (!is_depth_ushort && !is_depth_float) {
        PrintShaderWarning("Binding failed with incorrect depth image format.");
        return false;
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = 6;
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

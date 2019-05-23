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

#include "Open3D/Visualization/Shader/Simple2DShader.h"

#include "Open3D/Visualization/Shader/Shader.h"
#include "Open3D/Visualization/Utility/SelectionPolygon.h"
#include "Open3D/Visualization/Visualizer/RenderOptionWithEditing.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool Simple2DShader::Compile() {
    if (CompileShaders(Simple2DVertexShader, NULL, Simple2DFragmentShader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    return true;
}

void Simple2DShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool Simple2DShader::BindGeometry(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> colors;
    if (PrepareBinding(geometry, option, view, points, colors) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f),
                 colors.data(), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool Simple2DShader::RenderGeometry(const geometry::Geometry &geometry,
                                    const RenderOption &option,
                                    const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void Simple2DShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        bound_ = false;
    }
}

bool Simple2DShaderForSelectionPolygon::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Unspecified) {
        PrintShaderWarning("Rendering type is illegal.");
        return false;
    }
    glLineWidth(1.0f);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    return true;
}

bool Simple2DShaderForSelectionPolygon::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Unspecified) {
        PrintShaderWarning("Rendering type is illegal.");
        return false;
    }
    const SelectionPolygon &polygon = (const SelectionPolygon &)geometry;
    if (polygon.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty SelectionPolygon.");
    }
    size_t segment_num = polygon.polygon_.size() - 1;
    if (polygon.is_closed_) {
        segment_num++;
    }
    points.resize(segment_num * 2);
    colors.resize(segment_num * 2);
    for (size_t i = 0; i < segment_num; i++) {
    }
    if (polygon.is_closed_) {
        points.resize(polygon.polygon_.size() * 2);
        colors.resize(polygon.polygon_.size() * 2);
    } else {
        points.resize(polygon.polygon_.size() * 2 - 2);
        colors.resize(polygon.polygon_.size() * 2 - 2);
    }
    double width = (double)view.GetWindowWidth();
    double height = (double)view.GetWindowHeight();
    const auto &_option = (RenderOptionWithEditing &)option;
    for (size_t i = 0; i < segment_num; i++) {
        size_t j = (i + 1) % polygon.polygon_.size();
        const auto &vi = polygon.polygon_[i];
        const auto &vj = polygon.polygon_[j];
        points[i * 2] =
                Eigen::Vector3f((float)(vi(0) / width * 2.0 - 1.0),
                                (float)(vi(1) / height * 2.0 - 1.0), 0.0f);
        points[i * 2 + 1] =
                Eigen::Vector3f((float)(vj(0) / width * 2.0 - 1.0),
                                (float)(vj(1) / height * 2.0 - 1.0), 0.0f);
        colors[i * 2] = colors[i * 2 + 1] =
                _option.selection_polygon_boundary_color_.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

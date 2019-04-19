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

#include "Open3D/Visualization/Shader/CuboidShader.h"

#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Visualization/Shader/Shader.h"
#include "Open3D/Visualization/Utility/ColorMap.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool CuboidShader::Compile() {
    if (CompileShaders(SimpleVertexShader, NULL, SimpleFragmentShader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    line_vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    line_vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void CuboidShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool CuboidShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector3f> line_points;
    std::vector<Eigen::Vector3f> line_colors;
    if (PrepareBinding(geometry, option, view, points, colors, line_points,
                       line_colors) == false) {
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

    glGenBuffers(1, &line_vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, line_vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, line_points.size() * sizeof(Eigen::Vector3f),
                 line_points.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &line_vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, line_vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, line_colors.size() * sizeof(Eigen::Vector3f),
                 line_colors.data(), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool CuboidShader::RenderGeometry(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());

    if (option.mesh_show_wireframe_) {
        glEnableVertexAttribArray(line_vertex_position_);
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_position_buffer_);
        glVertexAttribPointer(line_vertex_position_, 3, GL_FLOAT, GL_FALSE, 0,
                              NULL);
        glEnableVertexAttribArray(line_vertex_color_);
        glBindBuffer(GL_ARRAY_BUFFER, line_vertex_color_buffer_);
        glVertexAttribPointer(line_vertex_color_, 3, GL_FLOAT, GL_FALSE, 0,
                              NULL);
        glDrawArrays(line_draw_arrays_mode_, 0, line_draw_arrays_size_);
        glDisableVertexAttribArray(line_vertex_position_);
        glDisableVertexAttribArray(line_vertex_color_);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnableVertexAttribArray(vertex_position_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
        glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(vertex_color_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
        glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
        glDisableVertexAttribArray(vertex_position_);
        glDisableVertexAttribArray(vertex_color_);
    }

    return true;
}

void CuboidShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        glDeleteBuffers(1, &line_vertex_position_buffer_);
        glDeleteBuffers(1, &line_vertex_color_buffer_);
        bound_ = false;
    }
}

bool CuboidShaderForVoxelGrid::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    return true;
}

bool CuboidShaderForVoxelGrid::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors,
        std::vector<Eigen::Vector3f> &line_points,
        std::vector<Eigen::Vector3f> &line_colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (voxel_grid.HasVoxels() == false) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    auto num_voxels = voxel_grid.voxels_.size();
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    // Templates for 8 points in a voxel
    std::vector<Eigen::Vector3i> vertex_offsets{
            Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
            Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(1, 1, 0),
            Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
            Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1),
    };

    // Templates for 12 triangles in a voxel, create right-handed manifold mesh
    std::vector<Eigen::Vector3i> triangles_vertex_indices{
            Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 1, 4),
            Eigen::Vector3i(0, 4, 2), Eigen::Vector3i(5, 1, 7),
            Eigen::Vector3i(5, 7, 4), Eigen::Vector3i(5, 4, 1),
            Eigen::Vector3i(3, 7, 1), Eigen::Vector3i(3, 1, 2),
            Eigen::Vector3i(3, 2, 7), Eigen::Vector3i(6, 4, 7),
            Eigen::Vector3i(6, 7, 2), Eigen::Vector3i(6, 2, 4),
    };

    std::vector<Eigen::Vector2i> lines_vertex_indices{
            Eigen::Vector2i(0, 1), Eigen::Vector2i(0, 2), Eigen::Vector2i(0, 4),
            Eigen::Vector2i(3, 1), Eigen::Vector2i(3, 2), Eigen::Vector2i(3, 7),
            Eigen::Vector2i(5, 1), Eigen::Vector2i(5, 4), Eigen::Vector2i(5, 7),
            Eigen::Vector2i(6, 2), Eigen::Vector2i(6, 4), Eigen::Vector2i(6, 7),
    };

    for (size_t voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                voxel_grid.origin_.cast<float>() +
                voxel_grid.voxels_[voxel_idx].cast<float>() *
                        voxel_grid.voxel_size_;
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : vertex_offsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<float>() *
                                                     voxel_grid.voxel_size_);
        }

        // Voxel color (applied to all points)
        Eigen::Vector3d voxel_color;
        switch (option.mesh_color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().GetXPercentage(base_vertex(0)));
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().GetYPercentage(base_vertex(1)));
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                voxel_color = global_color_map.GetColor(
                        view.GetBoundingBox().GetZPercentage(base_vertex(2)));
                break;
            case RenderOption::MeshColorOption::Color:
                if (voxel_grid.HasColors()) {
                    voxel_color = voxel_grid.colors_[voxel_idx];
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color = option.default_mesh_color_;
                break;
        }
        Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

        // 12 triangles in a voxel
        for (const Eigen::Vector3i &triangle_vertex_indices :
             triangles_vertex_indices) {
            points.push_back(vertices[triangle_vertex_indices(0)]);
            points.push_back(vertices[triangle_vertex_indices(1)]);
            points.push_back(vertices[triangle_vertex_indices(2)]);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
        }

        // 12 lines
        for (const Eigen::Vector2i &line_vertex_indices :
             lines_vertex_indices) {
            line_points.push_back(vertices[line_vertex_indices(0)]);
            line_points.push_back(vertices[line_vertex_indices(1)]);
            line_colors.push_back(voxel_color_f);
            line_colors.push_back(voxel_color_f);
        }
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());

    line_draw_arrays_mode_ = GL_LINES;
    line_draw_arrays_size_ = GLsizei(line_points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d

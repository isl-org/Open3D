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

#include "open3d/visualization/shader/SimpleShader.h"

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/Octree.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/visualization/shader/Shader.h"
#include "open3d/visualization/utility/ColorMap.h"

namespace open3d {
namespace visualization {
namespace glsl {

// Coordinates of 8 vertices in a cuboid (assume origin (0,0,0), size 1)
const static std::vector<Eigen::Vector3i> cuboid_vertex_offsets{
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(1, 1, 0),
        Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
        Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1),
};

// Vertex indices of 12 triangles in a cuboid, for right-handed manifold mesh
const static std::vector<Eigen::Vector3i> cuboid_triangles_vertex_indices{
        Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 1, 4),
        Eigen::Vector3i(0, 4, 2), Eigen::Vector3i(5, 1, 7),
        Eigen::Vector3i(5, 7, 4), Eigen::Vector3i(5, 4, 1),
        Eigen::Vector3i(3, 7, 1), Eigen::Vector3i(3, 1, 2),
        Eigen::Vector3i(3, 2, 7), Eigen::Vector3i(6, 4, 7),
        Eigen::Vector3i(6, 7, 2), Eigen::Vector3i(6, 2, 4),
};

// Vertex indices of 12 lines in a cuboid
const static std::vector<Eigen::Vector2i> cuboid_lines_vertex_indices{
        Eigen::Vector2i(0, 1), Eigen::Vector2i(0, 2), Eigen::Vector2i(0, 4),
        Eigen::Vector2i(3, 1), Eigen::Vector2i(3, 2), Eigen::Vector2i(3, 7),
        Eigen::Vector2i(5, 1), Eigen::Vector2i(5, 4), Eigen::Vector2i(5, 7),
        Eigen::Vector2i(6, 2), Eigen::Vector2i(6, 4), Eigen::Vector2i(6, 7),
};

bool SimpleShader::Compile() {
    if (!CompileShaders(SimpleVertexShader, NULL, SimpleFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleShader::BindGeometry(const geometry::Geometry &geometry,
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
    if (!PrepareBinding(geometry, option, view, points, colors)) {
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

bool SimpleShader::RenderGeometry(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
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

void SimpleShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        bound_ = false;
    }
}

bool SimpleShaderForPointCloud::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    const geometry::PointCloud &pointcloud =
            (const geometry::PointCloud &)geometry;
    if (!pointcloud.HasPoints()) {
        PrintShaderWarning("Binding failed with empty pointcloud.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.resize(pointcloud.points_.size());
    colors.resize(pointcloud.points_.size());
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const auto &point = pointcloud.points_[i];
        points[i] = point.cast<float>();
        Eigen::Vector3d color;
        switch (option.point_color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().GetXPercentage(point(0)));
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().GetYPercentage(point(1)));
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color = global_color_map.GetColor(
                        view.GetBoundingBox().GetZPercentage(point(2)));
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (pointcloud.HasColors()) {
                    color = pointcloud.colors_[i];
                } else {
                    color = global_color_map.GetColor(
                            view.GetBoundingBox().GetZPercentage(point(2)));
                }
                break;
        }
        colors[i] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForLineSet::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForLineSet::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    const geometry::LineSet &lineset = (const geometry::LineSet &)geometry;
    if (!lineset.HasLines()) {
        PrintShaderWarning("Binding failed with empty geometry::LineSet.");
        return false;
    }
    points.resize(lineset.lines_.size() * 2);
    colors.resize(lineset.lines_.size() * 2);
    for (size_t i = 0; i < lineset.lines_.size(); i++) {
        const auto point_pair = lineset.GetLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset.HasColors()) {
            color = lineset.colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForTetraMesh::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TetraMesh) {
        PrintShaderWarning("Rendering type is not geometry::TetraMesh.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForTetraMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    typedef decltype(geometry::TetraMesh::tetras_)::value_type TetraIndices;
    typedef decltype(geometry::TetraMesh::tetras_)::value_type::Scalar Index;
    typedef std::tuple<Index, Index> Index2;

    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TetraMesh) {
        PrintShaderWarning("Rendering type is not geometry::TetraMesh.");
        return false;
    }
    const geometry::TetraMesh &tetramesh =
            (const geometry::TetraMesh &)geometry;
    if (!tetramesh.HasTetras()) {
        PrintShaderWarning("Binding failed with empty geometry::TetraMesh.");
        return false;
    }

    std::unordered_set<Index2, utility::hash_tuple<Index2>> inserted_edges;
    auto InsertEdge = [&](Index vidx0, Index vidx1) {
        Index2 edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            Eigen::Vector3f p0 = tetramesh.vertices_[vidx0].cast<float>();
            Eigen::Vector3f p1 = tetramesh.vertices_[vidx1].cast<float>();
            points.insert(points.end(), {p0, p1});
            Eigen::Vector3f color(0, 0, 0);
            colors.insert(colors.end(), {color, color});
        }
    };

    for (size_t i = 0; i < tetramesh.tetras_.size(); i++) {
        const TetraIndices tetra = tetramesh.tetras_[i];
        InsertEdge(tetra(0), tetra(1));
        InsertEdge(tetra(1), tetra(2));
        InsertEdge(tetra(2), tetra(0));
        InsertEdge(tetra(3), tetra(0));
        InsertEdge(tetra(3), tetra(1));
        InsertEdge(tetra(3), tetra(2));
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForOrientedBoundingBox::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::OrientedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::OrientedBoundingBox.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOrientedBoundingBox::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::OrientedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::OrientedBoundingBox.");
        return false;
    }
    auto lineset = geometry::LineSet::CreateFromOrientedBoundingBox(
            (const geometry::OrientedBoundingBox &)geometry);
    points.resize(lineset->lines_.size() * 2);
    colors.resize(lineset->lines_.size() * 2);
    for (size_t i = 0; i < lineset->lines_.size(); i++) {
        const auto point_pair = lineset->GetLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset->HasColors()) {
            color = lineset->colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::AxisAlignedBoundingBox.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::AxisAlignedBoundingBox.");
        return false;
    }
    auto lineset = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            (const geometry::AxisAlignedBoundingBox &)geometry);
    points.resize(lineset->lines_.size() * 2);
    colors.resize(lineset->lines_.size() * 2);
    for (size_t i = 0; i < lineset->lines_.size(); i++) {
        const auto point_pair = lineset->GetLineCoordinate(i);
        points[i * 2] = point_pair.first.cast<float>();
        points[i * 2 + 1] = point_pair.second.cast<float>();
        Eigen::Vector3d color;
        if (lineset->HasColors()) {
            color = lineset->colors_[i];
        } else {
            color = Eigen::Vector3d::Zero();
        }
        colors[i * 2] = colors[i * 2 + 1] = color.cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForTriangleMesh::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool SimpleShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh &&
        geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::HalfEdgeTriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    const geometry::TriangleMesh &mesh =
            (const geometry::TriangleMesh &)geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.resize(mesh.triangles_.size() * 3);
    colors.resize(mesh.triangles_.size() * 3);

    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t vi = triangle(j);
            const auto &vertex = mesh.vertices_[vi];
            points[idx] = vertex.cast<float>();

            Eigen::Vector3d color;
            switch (option.mesh_color_option_) {
                case RenderOption::MeshColorOption::XCoordinate:
                    color = global_color_map.GetColor(
                            view.GetBoundingBox().GetXPercentage(vertex(0)));
                    break;
                case RenderOption::MeshColorOption::YCoordinate:
                    color = global_color_map.GetColor(
                            view.GetBoundingBox().GetYPercentage(vertex(1)));
                    break;
                case RenderOption::MeshColorOption::ZCoordinate:
                    color = global_color_map.GetColor(
                            view.GetBoundingBox().GetZPercentage(vertex(2)));
                    break;
                case RenderOption::MeshColorOption::Color:
                    if (mesh.HasVertexColors()) {
                        color = mesh.vertex_colors_[vi];
                        break;
                    }
                case RenderOption::MeshColorOption::Default:
                default:
                    color = option.default_mesh_color_;
                    break;
            }
            colors[idx] = color.cast<float>();
        }
    }
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForVoxelGridLine::PrepareRendering(
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
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForVoxelGridLine::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (!voxel_grid.HasVoxels()) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 12 * 2
    colors.clear();  // Final size: num_voxels * 12 * 2

    for (auto &it : voxel_grid.voxels_) {
        const geometry::Voxel &voxel = it.second;
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                voxel_grid.origin_.cast<float>() +
                voxel.grid_index_.cast<float>() * voxel_grid.voxel_size_;
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
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
                    voxel_color = voxel.color_;
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color = option.default_mesh_color_;
                break;
        }
        Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

        // 12 lines
        for (const Eigen::Vector2i &line_vertex_indices :
             cuboid_lines_vertex_indices) {
            points.push_back(vertices[line_vertex_indices(0)]);
            points.push_back(vertices[line_vertex_indices(1)]);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
        }
    }

    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForVoxelGridFace::PrepareRendering(
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
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForVoxelGridFace::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (!voxel_grid.HasVoxels()) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    for (auto &it : voxel_grid.voxels_) {
        const geometry::Voxel &voxel = it.second;
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                voxel_grid.origin_.cast<float>() +
                voxel.grid_index_.cast<float>() * voxel_grid.voxel_size_;
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
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
                    voxel_color = voxel.color_;
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
             cuboid_triangles_vertex_indices) {
            points.push_back(vertices[triangle_vertex_indices(0)]);
            points.push_back(vertices[triangle_vertex_indices(1)]);
            points.push_back(vertices[triangle_vertex_indices(2)]);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
            colors.push_back(voxel_color_f);
        }
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

bool SimpleShaderForOctreeFace::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Octree) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOctreeFace::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Octree) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    const geometry::Octree &octree = (const geometry::Octree &)geometry;
    if (octree.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty octree.");
        return false;
    }
    const ColorMap &global_color_map = *GetGlobalColorMap();
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    auto f = [&points, &colors, &option, &global_color_map, &view](
                     const std::shared_ptr<geometry::OctreeNode> &node,
                     const std::shared_ptr<geometry::OctreeNodeInfo> &node_info)
            -> bool {
        if (auto leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                            node)) {
            // All vertex in the voxel share the same color
            Eigen::Vector3f base_vertex = node_info->origin_.cast<float>();
            std::vector<Eigen::Vector3f> vertices;
            for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
                vertices.push_back(base_vertex +
                                   vertex_offset.cast<float>() *
                                           float(node_info->size_));
            }

            Eigen::Vector3d voxel_color;
            switch (option.mesh_color_option_) {
                case RenderOption::MeshColorOption::XCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().GetXPercentage(
                                    base_vertex(0)));
                    break;
                case RenderOption::MeshColorOption::YCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().GetYPercentage(
                                    base_vertex(1)));
                    break;
                case RenderOption::MeshColorOption::ZCoordinate:
                    voxel_color = global_color_map.GetColor(
                            view.GetBoundingBox().GetZPercentage(
                                    base_vertex(2)));
                    break;
                case RenderOption::MeshColorOption::Color:
                    voxel_color = leaf_node->color_;
                    break;
                case RenderOption::MeshColorOption::Default:
                default:
                    voxel_color = option.default_mesh_color_;
                    break;
            }
            Eigen::Vector3f voxel_color_f = voxel_color.cast<float>();

            // 12 triangles in a voxel
            for (const Eigen::Vector3i &triangle_vertex_indices :
                 cuboid_triangles_vertex_indices) {
                points.push_back(vertices[triangle_vertex_indices(0)]);
                points.push_back(vertices[triangle_vertex_indices(1)]);
                points.push_back(vertices[triangle_vertex_indices(2)]);
                colors.push_back(voxel_color_f);
                colors.push_back(voxel_color_f);
                colors.push_back(voxel_color_f);
            }
        }
        return false;
    };

    octree.Traverse(f);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

bool SimpleShaderForOctreeLine::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Octree) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForOctreeLine::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::Octree) {
        PrintShaderWarning("Rendering type is not geometry::Octree.");
        return false;
    }
    const geometry::Octree &octree = (const geometry::Octree &)geometry;
    if (octree.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty octree.");
        return false;
    }
    points.clear();  // Final size: num_voxels * 36
    colors.clear();  // Final size: num_voxels * 36

    auto f = [&points, &colors](
                     const std::shared_ptr<geometry::OctreeNode> &node,
                     const std::shared_ptr<geometry::OctreeNodeInfo> &node_info)
            -> bool {
        Eigen::Vector3f base_vertex = node_info->origin_.cast<float>();
        std::vector<Eigen::Vector3f> vertices;
        for (const Eigen::Vector3i &vertex_offset : cuboid_vertex_offsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<float>() *
                                                     float(node_info->size_));
        }
        Eigen::Vector3f voxel_color = Eigen::Vector3f::Zero();
        if (auto leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                            node)) {
            voxel_color = leaf_node->color_.cast<float>();
        }

        for (const Eigen::Vector2i &line_vertex_indices :
             cuboid_lines_vertex_indices) {
            points.push_back(vertices[line_vertex_indices(0)]);
            points.push_back(vertices[line_vertex_indices(1)]);
            colors.push_back(voxel_color);
            colors.push_back(voxel_color);
        }
        return false;
    };

    octree.Traverse(f);

    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());

    return true;
}

}  // namespace glsl
}  // namespace visualization
}  // namespace open3d

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

#include <rply/rply.h>

#include "Open3D/IO/ClassIO/LineSetIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/IO/ClassIO/VoxelGridIO.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

namespace {
using namespace io;

namespace ply_pointcloud_reader {

struct PLYReaderState {
    utility::ConsoleProgressBar *progress_bar;
    geometry::PointCloud *pointcloud_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;  // some sanity check
    }

    double value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->points_[state_ptr->vertex_index](index) = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->normals_[state_ptr->normal_index](index) = value;
    if (index == 2) {  // reading 'nz'
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->colors_[state_ptr->color_index](index) =
            value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}

}  // namespace ply_pointcloud_reader

namespace ply_trianglemesh_reader {

struct PLYReaderState {
    utility::ConsoleProgressBar *progress_bar;
    geometry::TriangleMesh *mesh_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;
    std::vector<unsigned int> face;
    long face_index;
    long face_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertices_[state_ptr->vertex_index](index) = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_normals_[state_ptr->normal_index](index) =
            value;
    if (index == 2) {  // reading 'nz'
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_colors_[state_ptr->color_index](index) =
            value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}

int ReadFaceCallBack(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long dummy, length, index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &dummy);
    double value = ply_get_argument_value(argument);
    if (state_ptr->face_index >= state_ptr->face_num) {
        return 0;
    }

    ply_get_argument_property(argument, NULL, &length, &index);
    if (index == -1) {
        state_ptr->face.clear();
    } else {
        state_ptr->face.push_back(int(value));
    }
    if (long(state_ptr->face.size()) == length) {
        if (!AddTrianglesByEarClipping(*state_ptr->mesh_ptr, state_ptr->face)) {
            utility::LogWarning(
                    "Read PLY failed: A polygon in the mesh could not be "
                    "decomposed into triangles.\n");
            return 0;
        }
        state_ptr->face_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

}  // namespace ply_trianglemesh_reader

namespace ply_lineset_reader {

struct PLYReaderState {
    utility::ConsoleProgressBar *progress_bar;
    geometry::LineSet *lineset_ptr;
    long vertex_index;
    long vertex_num;
    long line_index;
    long line_num;
    long color_index;
    long color_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;  // some sanity check
    }

    double value = ply_get_argument_value(argument);
    state_ptr->lineset_ptr->points_[state_ptr->vertex_index](index) = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadLineCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->line_index >= state_ptr->line_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->lineset_ptr->lines_[state_ptr->line_index](index) = int(value);
    if (index == 1) {  // reading 'vertex2'
        state_ptr->line_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->lineset_ptr->colors_[state_ptr->color_index](index) =
            value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

}  // namespace ply_lineset_reader

namespace ply_voxelgrid_reader {

struct PLYReaderState {
    utility::ConsoleProgressBar *progress_bar;
    geometry::VoxelGrid *voxelgrid_ptr;
    long voxel_index;
    long voxel_num;
    long color_index;
    long color_num;
};

int ReadOriginCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);

    double value = ply_get_argument_value(argument);
    state_ptr->voxelgrid_ptr->origin_(index) = value;
    return 1;
}

int ReadScaleCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);

    double value = ply_get_argument_value(argument);
    state_ptr->voxelgrid_ptr->voxel_size_ = value;
    return 1;
}

int ReadVoxelCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->voxel_index >= state_ptr->voxel_num) {
        return 0;  // some sanity check
    }

    double value = ply_get_argument_value(argument);
    state_ptr->voxelgrid_ptr->voxels_[state_ptr->voxel_index].grid_index_(
            index) = int(value);
    if (index == 2) {  // reading 'z'
        state_ptr->voxel_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->voxelgrid_ptr->voxels_[state_ptr->color_index].color_(index) =
            value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

}  // namespace ply_voxelgrid_reader

}  // unnamed namespace

namespace io {

bool ReadPointCloudFromPLY(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           bool print_progress) {
    using namespace ply_pointcloud_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: %s\n",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.\n");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.pointcloud_ptr = &pointcloud;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.\n");
        ply_close(ply_file);
        return false;
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;

    pointcloud.Clear();
    pointcloud.points_.resize(state.vertex_num);
    pointcloud.normals_.resize(state.normal_num);
    pointcloud.colors_.resize(state.color_num);

    utility::ConsoleProgressBar progress_bar(state.vertex_num + 1,
                                             "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}\n",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    ++progress_bar;
    return true;
}

bool WritePointCloudToPLY(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          bool write_ascii /* = false*/,
                          bool compressed /* = false*/,
                          bool print_progress) {
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write PLY failed: point cloud has 0 points.\n");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(pointcloud.points_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (pointcloud.HasNormals()) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (pointcloud.HasColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.\n");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(pointcloud.points_.size()),
            "Writing PLY: ", print_progress);

    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const Eigen::Vector3d &point = pointcloud.points_[i];
        ply_write(ply_file, point(0));
        ply_write(ply_file, point(1));
        ply_write(ply_file, point(2));
        if (pointcloud.HasNormals()) {
            const Eigen::Vector3d &normal = pointcloud.normals_[i];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (pointcloud.HasColors()) {
            const Eigen::Vector3d &color = pointcloud.colors_[i];
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(0) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(1) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(2) * 255.0)));
        }
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

bool ReadTriangleMeshFromPLY(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             bool print_progress) {
    using namespace ply_trianglemesh_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.\n");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.mesh_ptr = &mesh;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.\n");
        ply_close(ply_file);
        return false;
    }

    state.face_num = ply_set_read_cb(ply_file, "face", "vertex_indices",
                                     ReadFaceCallBack, &state, 0);
    if (state.face_num == 0) {
        state.face_num = ply_set_read_cb(ply_file, "face", "vertex_index",
                                         ReadFaceCallBack, &state, 0);
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;
    state.face_index = 0;

    mesh.Clear();
    mesh.vertices_.resize(state.vertex_num);
    mesh.vertex_normals_.resize(state.normal_num);
    mesh.vertex_colors_.resize(state.color_num);

    utility::ConsoleProgressBar progress_bar(state.vertex_num + state.face_num,
                                             "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}\n",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool WriteTriangleMeshToPLY(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool print_progress) {
    if (mesh.IsEmpty()) {
        utility::LogWarning("Write PLY failed: mesh has 0 vertices.\n");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }

    write_vertex_normals = write_vertex_normals && mesh.HasVertexNormals();
    write_vertex_colors = write_vertex_colors && mesh.HasVertexColors();

    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(mesh.vertices_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (write_vertex_normals) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (write_vertex_colors) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    ply_add_element(ply_file, "face",
                    static_cast<long>(mesh.triangles_.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.\n");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(mesh.vertices_.size() + mesh.triangles_.size()),
            "Writing PLY: ", print_progress);
    for (size_t i = 0; i < mesh.vertices_.size(); i++) {
        const auto &vertex = mesh.vertices_[i];
        ply_write(ply_file, vertex(0));
        ply_write(ply_file, vertex(1));
        ply_write(ply_file, vertex(2));
        if (write_vertex_normals) {
            const auto &normal = mesh.vertex_normals_[i];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (write_vertex_colors) {
            const auto &color = mesh.vertex_colors_[i];
            ply_write(ply_file, color(0) * 255.0);
            ply_write(ply_file, color(1) * 255.0);
            ply_write(ply_file, color(2) * 255.0);
        }
        ++progress_bar;
    }
    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle(0));
        ply_write(ply_file, triangle(1));
        ply_write(ply_file, triangle(2));
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

bool ReadLineSetFromPLY(const std::string &filename,
                        geometry::LineSet &lineset,
                        bool print_progress) {
    using namespace ply_lineset_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.\n");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.lineset_ptr = &lineset;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.line_num = ply_set_read_cb(ply_file, "edge", "vertex1",
                                     ReadLineCallback, &state, 0);
    ply_set_read_cb(ply_file, "edge", "vertex2", ReadLineCallback, &state, 1);

    state.color_num = ply_set_read_cb(ply_file, "edge", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "edge", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "edge", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.\n");
        ply_close(ply_file);
        return false;
    }
    if (state.line_num <= 0) {
        utility::LogWarning("Read PLY failed: number of edges <= 0.\n");
        ply_close(ply_file);
        return false;
    }

    state.vertex_index = 0;
    state.line_index = 0;
    state.color_index = 0;

    lineset.Clear();
    lineset.points_.resize(state.vertex_num);
    lineset.lines_.resize(state.line_num);
    lineset.colors_.resize(state.color_num);

    utility::ConsoleProgressBar progress_bar(
            state.vertex_num + state.line_num + state.color_num,
            "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}\n",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool WriteLineSetToPLY(const std::string &filename,
                       const geometry::LineSet &lineset,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/,
                       bool print_progress) {
    if (lineset.IsEmpty()) {
        utility::LogWarning("Write PLY failed: line set has 0 points.\n");
        return false;
    }
    if (!lineset.HasLines()) {
        utility::LogWarning("Write PLY failed: line set has 0 lines.\n");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(lineset.points_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_element(ply_file, "edge", static_cast<long>(lineset.lines_.size()));
    ply_add_property(ply_file, "vertex1", PLY_INT, PLY_INT, PLY_INT);
    ply_add_property(ply_file, "vertex2", PLY_INT, PLY_INT, PLY_INT);
    if (lineset.HasColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.\n");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(lineset.points_.size() + lineset.lines_.size()),
            "Writing PLY: ", print_progress);

    for (size_t i = 0; i < lineset.points_.size(); i++) {
        const Eigen::Vector3d &point = lineset.points_[i];
        ply_write(ply_file, point(0));
        ply_write(ply_file, point(1));
        ply_write(ply_file, point(2));
        ++progress_bar;
    }
    for (size_t i = 0; i < lineset.lines_.size(); i++) {
        const Eigen::Vector2i &line = lineset.lines_[i];
        ply_write(ply_file, line(0));
        ply_write(ply_file, line(1));
        if (lineset.HasColors()) {
            const Eigen::Vector3d &color = lineset.colors_[i];
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(0) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(1) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(2) * 255.0)));
        }
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

bool ReadVoxelGridFromPLY(const std::string &filename,
                          geometry::VoxelGrid &voxelgrid,
                          bool print_progress) {
    using namespace ply_voxelgrid_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.\n");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.voxelgrid_ptr = &voxelgrid;
    state.voxel_num = ply_set_read_cb(ply_file, "vertex", "x",
                                      ReadVoxelCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVoxelCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVoxelCallback, &state, 2);

    if (state.voxel_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.\n");
        ply_close(ply_file);
        return false;
    }

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    ply_set_read_cb(ply_file, "origin", "x", ReadOriginCallback, &state, 0);
    ply_set_read_cb(ply_file, "origin", "y", ReadOriginCallback, &state, 1);
    ply_set_read_cb(ply_file, "origin", "z", ReadOriginCallback, &state, 2);
    ply_set_read_cb(ply_file, "voxel_size", "val", ReadScaleCallback, &state,
                    0);

    state.voxel_index = 0;
    state.color_index = 0;

    voxelgrid.Clear();
    voxelgrid.voxels_.resize(state.voxel_num);

    utility::ConsoleProgressBar progress_bar(state.voxel_num + state.color_num,
                                             "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}\n",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool WriteVoxelGridToPLY(const std::string &filename,
                         const geometry::VoxelGrid &voxelgrid,
                         bool write_ascii /* = false*/,
                         bool compressed /* = false*/,
                         bool print_progress) {
    if (voxelgrid.IsEmpty()) {
        utility::LogWarning("Write PLY failed: voxelgrid has 0 voxels.\n");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "origin", 1);
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_element(ply_file, "voxel_size", 1);
    ply_add_property(ply_file, "val", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);

    ply_add_element(ply_file, "vertex",
                    static_cast<long>(voxelgrid.voxels_.size()));
    // PLY_UINT could be used for x, y, z but PLY_DOUBLE used instead due to
    // compatibility issue.
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (voxelgrid.HasColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }

    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.\n");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(voxelgrid.voxels_.size()),
            "Writing PLY: ", print_progress);

    const Eigen::Vector3d &origin = voxelgrid.origin_;
    ply_write(ply_file, origin(0));
    ply_write(ply_file, origin(1));
    ply_write(ply_file, origin(2));
    ply_write(ply_file, voxelgrid.voxel_size_);

    for (size_t i = 0; i < voxelgrid.voxels_.size(); i++) {
        const geometry::Voxel &voxel = voxelgrid.voxels_[i];
        ply_write(ply_file, voxel.grid_index_(0));
        ply_write(ply_file, voxel.grid_index_(1));
        ply_write(ply_file, voxel.grid_index_(2));

        const Eigen::Vector3d &color = voxel.color_;
        ply_write(ply_file, std::min(255.0, std::max(0.0, color(0) * 255.0)));
        ply_write(ply_file, std::min(255.0, std::max(0.0, color(1) * 255.0)));
        ply_write(ply_file, std::min(255.0, std::max(0.0, color(2) * 255.0)));

        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

}  // namespace io
}  // namespace open3d

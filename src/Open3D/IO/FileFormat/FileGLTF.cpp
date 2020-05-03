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

#include <numeric>
#include <vector>

#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

#include <tiny_gltf.h>

namespace open3d {
namespace io {

// Adapts an array of bytes to an array of T. Will advance of byte_stride each
// elements.
template <typename T>
struct ArrayAdapter {
    // Pointer to the bytes
    const unsigned char* data_ptr;
    // Number of elements in the array
    const size_t elem_count;
    // Stride in bytes between two elements
    const size_t stride;

    // Construct an array adapter.
    // \param ptr Pointer to the start of the data, with offset applied
    // \param count Number of elements in the array
    // \param byte_stride Stride betweens elements in the array
    ArrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
        : data_ptr(ptr), elem_count(count), stride(byte_stride) {}

    // Returns a *copy* of a single element. Can't be used to modify it.
    T operator[](size_t pos) const {
        if (pos >= elem_count)
            throw std::out_of_range(
                    "Tried to access beyond the last element of an array "
                    "adapter with count " +
                    std::to_string(elem_count) +
                    " while getting element number " + std::to_string(pos));
        return *(reinterpret_cast<const T*>(data_ptr + pos * stride));
    }
};

// Interface of any adapted array that returns integer data
struct IntArrayBase {
    virtual ~IntArrayBase() = default;
    virtual unsigned int operator[](size_t) const = 0;
    virtual size_t size() const = 0;
};

// An array that loads integer types, and returns them as int
template <class T>
struct IntArray : public IntArrayBase {
    ArrayAdapter<T> adapter;

    IntArray(const ArrayAdapter<T>& a) : adapter(a) {}
    unsigned int operator[](size_t position) const override {
        return static_cast<unsigned int>(adapter[position]);
    }

    size_t size() const override { return adapter.elem_count; }
};

bool ReadTriangleMeshFromGLTF(const std::string& filename,
                              geometry::TriangleMesh& mesh,
                              bool print_progress) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn;
    std::string err;

    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    bool ret;
    if (filename_ext == "glb") {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
    }

    if (!warn.empty() || !err.empty()) {
        utility::LogWarning("Read GLTF failed: unable to open file {}",
                            filename);
    }
    if (!ret) {
        return false;
    }

    if (model.meshes.size() > 1) {
        utility::LogInfo(
                "The file contains more than one mesh. All meshes will be "
                "loaded as a single mesh.");
    }

    mesh.Clear();
    geometry::TriangleMesh mesh_temp;
    for (const tinygltf::Node& gltf_node : model.nodes) {
        if (gltf_node.mesh != -1) {
            mesh_temp.Clear();
            const tinygltf::Mesh& gltf_mesh = model.meshes[gltf_node.mesh];

            for (const tinygltf::Primitive& primitive : gltf_mesh.primitives) {
                for (const auto& attribute : primitive.attributes) {
                    if (attribute.first == "POSITION") {
                        tinygltf::Accessor& positions_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& positions_view =
                                model.bufferViews[positions_accessor
                                                          .bufferView];
                        const tinygltf::Buffer& positions_buffer =
                                model.buffers[positions_view.buffer];
                        const float* positions = reinterpret_cast<const float*>(
                                &positions_buffer
                                         .data[positions_view.byteOffset +
                                               positions_accessor.byteOffset]);

                        for (size_t i = 0; i < positions_accessor.count; ++i) {
                            mesh_temp.vertices_.push_back(Eigen::Vector3d(
                                    positions[i * 3 + 0], positions[i * 3 + 1],
                                    positions[i * 3 + 2]));
                        }
                    }

                    if (attribute.first == "NORMAL") {
                        tinygltf::Accessor& normals_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& normals_view =
                                model.bufferViews[normals_accessor.bufferView];
                        const tinygltf::Buffer& normals_buffer =
                                model.buffers[normals_view.buffer];
                        const float* normals = reinterpret_cast<const float*>(
                                &normals_buffer
                                         .data[normals_view.byteOffset +
                                               normals_accessor.byteOffset]);

                        for (size_t i = 0; i < normals_accessor.count; ++i) {
                            mesh_temp.vertex_normals_.push_back(Eigen::Vector3d(
                                    normals[i * 3 + 0], normals[i * 3 + 1],
                                    normals[i * 3 + 2]));
                        }
                    }

                    if (attribute.first == "COLOR_0") {
                        tinygltf::Accessor& colors_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& colors_view =
                                model.bufferViews[colors_accessor.bufferView];
                        const tinygltf::Buffer& colors_buffer =
                                model.buffers[colors_view.buffer];

                        size_t byte_stride = colors_view.byteStride;
                        if (byte_stride == 0) {
                            // According to glTF 2.0 specs:
                            // When byteStride==0, it means that accessor
                            // elements are tightly packed.
                            byte_stride =
                                    colors_accessor.type *
                                    tinygltf::GetComponentSizeInBytes(
                                            colors_accessor.componentType);
                        }
                        switch (colors_accessor.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const float* colors =
                                            reinterpret_cast<const float*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);
                                    mesh_temp.vertex_colors_.emplace_back(
                                            colors[0], colors[1], colors[2]);
                                }
                                break;
                            }
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                                double max_val = (double)
                                        std::numeric_limits<uint8_t>::max();
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const uint8_t* colors =
                                            reinterpret_cast<const uint8_t*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);
                                    mesh_temp.vertex_colors_.emplace_back(
                                            colors[0] / max_val,
                                            colors[1] / max_val,
                                            colors[2] / max_val);
                                }
                                break;
                            }
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                                double max_val = (double)
                                        std::numeric_limits<uint16_t>::max();
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const uint16_t* colors =
                                            reinterpret_cast<const uint16_t*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);
                                    mesh_temp.vertex_colors_.emplace_back(
                                            colors[0] / max_val,
                                            colors[1] / max_val,
                                            colors[2] / max_val);
                                }
                                break;
                            }
                            default: {
                                utility::LogWarning(
                                        "Unrecognized component type for "
                                        "vertex colors");
                                break;
                            }
                        }
                    }
                }

                // Load triangles
                std::unique_ptr<IntArrayBase> indices_array_pointer = nullptr;
                {
                    const tinygltf::Accessor& indices_accessor =
                            model.accessors[primitive.indices];
                    const tinygltf::BufferView& indices_view =
                            model.bufferViews[indices_accessor.bufferView];
                    const tinygltf::Buffer& indices_buffer =
                            model.buffers[indices_view.buffer];
                    const auto data_address = indices_buffer.data.data() +
                                              indices_view.byteOffset +
                                              indices_accessor.byteOffset;
                    const auto byte_stride =
                            indices_accessor.ByteStride(indices_view);
                    const auto count = indices_accessor.count;

                    // Allocate the index array in the pointer-to-base
                    // declared in the parent scope
                    switch (indices_accessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_BYTE:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<char>>(
                                            new IntArray<char>(
                                                    ArrayAdapter<char>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<unsigned char>>(
                                            new IntArray<unsigned char>(
                                                    ArrayAdapter<unsigned char>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_SHORT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<short>>(
                                            new IntArray<short>(
                                                    ArrayAdapter<short>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indices_array_pointer = std::unique_ptr<
                                    IntArray<unsigned short>>(
                                    new IntArray<unsigned short>(
                                            ArrayAdapter<unsigned short>(
                                                    data_address, count,
                                                    byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_INT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<int>>(
                                            new IntArray<int>(ArrayAdapter<int>(
                                                    data_address, count,
                                                    byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<unsigned int>>(
                                            new IntArray<unsigned int>(
                                                    ArrayAdapter<unsigned int>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        default:
                            break;
                    }
                    const auto& indices = *indices_array_pointer;

                    switch (primitive.mode) {
                        case TINYGLTF_MODE_TRIANGLES:
                            for (size_t i = 0; i < indices_accessor.count;
                                 i += 3) {
                                mesh_temp.triangles_.push_back(Eigen::Vector3i(
                                        indices[i], indices[i + 1],
                                        indices[i + 2]));
                            }
                            break;
                        case TINYGLTF_MODE_TRIANGLE_STRIP:
                            for (size_t i = 2; i < indices_accessor.count;
                                 ++i) {
                                mesh_temp.triangles_.push_back(Eigen::Vector3i(
                                        indices[i - 2], indices[i - 1],
                                        indices[i]));
                            }
                            break;
                        case TINYGLTF_MODE_TRIANGLE_FAN:
                            for (size_t i = 2; i < indices_accessor.count;
                                 ++i) {
                                mesh_temp.triangles_.push_back(Eigen::Vector3i(
                                        indices[0], indices[i - 1],
                                        indices[i]));
                            }
                            break;
                    }
                }
            }

            if (gltf_node.matrix.size() > 0) {
                std::vector<double> matrix = gltf_node.matrix;
                Eigen::Matrix4d transform =
                        Eigen::Map<Eigen::Matrix4d>(&matrix[0], 4, 4);
                mesh_temp.Transform(transform);
            } else {
                // The specification states that first the scale is
                // applied to the vertices, then the rotation, and then the
                // translation.
                if (gltf_node.scale.size() > 0) {
                    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                    transform(0, 0) = gltf_node.scale[0];
                    transform(1, 1) = gltf_node.scale[1];
                    transform(2, 2) = gltf_node.scale[2];
                    mesh_temp.Transform(transform);
                }
                if (gltf_node.rotation.size() > 0) {
                    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                    // glTF represents a quaternion as qx, qy, qz, qw, while
                    // Eigen::Quaterniond orders the parameters as qw, qx,
                    // qy, qz.
                    transform.topLeftCorner<3, 3>() =
                            Eigen::Quaterniond(gltf_node.rotation[3],
                                               gltf_node.rotation[0],
                                               gltf_node.rotation[1],
                                               gltf_node.rotation[2])
                                    .toRotationMatrix();
                    mesh_temp.Transform(transform);
                }
                if (gltf_node.translation.size() > 0) {
                    mesh_temp.Translate(Eigen::Vector3d(
                            gltf_node.translation[0], gltf_node.translation[1],
                            gltf_node.translation[2]));
                }
            }
            mesh += mesh_temp;
        }
    }

    return true;
}

bool WriteTriangleMeshToGLTF(const std::string& filename,
                             const geometry::TriangleMesh& mesh,
                             bool write_ascii /* = false*/,
                             bool compressed /* = false*/,
                             bool write_vertex_normals /* = true*/,
                             bool write_vertex_colors /* = true*/,
                             bool write_triangle_uvs /* = true*/,
                             bool print_progress) {
    if (write_triangle_uvs && mesh.HasTriangleUvs()) {
        utility::LogWarning(
                "This file format does not support writing textures and uv "
                "coordinates. Consider using .obj");
    }
    tinygltf::Model model;
    model.asset.generator = "Open3D";
    model.asset.version = "2.0";
    model.defaultScene = 0;

    size_t byte_length;
    size_t num_of_vertices = mesh.vertices_.size();
    size_t num_of_triangles = mesh.triangles_.size();

    float float_temp;
    unsigned char* temp = NULL;

    tinygltf::BufferView indices_buffer_view_array;
    bool save_indices_as_uint32 = num_of_vertices > 65536;
    indices_buffer_view_array.name = save_indices_as_uint32
                                             ? "buffer-0-bufferview-uint"
                                             : "buffer-0-bufferview-ushort";
    indices_buffer_view_array.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    indices_buffer_view_array.buffer = 0;
    indices_buffer_view_array.byteLength = 0;
    model.bufferViews.push_back(indices_buffer_view_array);
    size_t indices_buffer_view_index = model.bufferViews.size() - 1;

    tinygltf::BufferView buffer_view_array;
    buffer_view_array.name = "buffer-0-bufferview-vec3",
    buffer_view_array.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    buffer_view_array.buffer = 0;
    buffer_view_array.byteLength = 0;
    buffer_view_array.byteOffset = 0;
    buffer_view_array.byteStride = 12;
    model.bufferViews.push_back(buffer_view_array);
    size_t mesh_attributes_buffer_view_index = model.bufferViews.size() - 1;

    tinygltf::Scene gltf_scene;
    gltf_scene.nodes.push_back(0);
    model.scenes.push_back(gltf_scene);

    tinygltf::Node gltf_node;
    gltf_node.mesh = 0;
    model.nodes.push_back(gltf_node);

    tinygltf::Mesh gltf_mesh;
    tinygltf::Primitive gltf_primitive;

    tinygltf::Accessor indices_accessor;
    indices_accessor.name = "buffer-0-accessor-indices-buffer-0-mesh-0";
    indices_accessor.type = TINYGLTF_TYPE_SCALAR;
    indices_accessor.componentType =
            save_indices_as_uint32 ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
                                   : TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
    indices_accessor.count = 3 * num_of_triangles;
    byte_length =
            3 * num_of_triangles *
            (save_indices_as_uint32 ? sizeof(uint32_t) : sizeof(uint16_t));

    indices_accessor.bufferView = int(indices_buffer_view_index);
    indices_accessor.byteOffset =
            model.bufferViews[indices_buffer_view_index].byteLength;
    model.bufferViews[indices_buffer_view_index].byteLength += byte_length;

    std::vector<unsigned char> index_buffer;
    for (size_t tidx = 0; tidx < num_of_triangles; ++tidx) {
        const Eigen::Vector3i& triangle = mesh.triangles_[tidx];
        size_t uint_size =
                save_indices_as_uint32 ? sizeof(uint32_t) : sizeof(uint16_t);
        for (size_t i = 0; i < 3; ++i) {
            temp = (unsigned char*)&(triangle(i));
            for (size_t j = 0; j < uint_size; ++j) {
                index_buffer.push_back(temp[j]);
            }
        }
    }

    indices_accessor.minValues.push_back(0);
    indices_accessor.maxValues.push_back(3 * int(num_of_triangles) - 1);
    model.accessors.push_back(indices_accessor);
    gltf_primitive.indices = int(model.accessors.size()) - 1;

    tinygltf::Accessor positions_accessor;
    positions_accessor.name = "buffer-0-accessor-position-buffer-0-mesh-0";
    positions_accessor.type = TINYGLTF_TYPE_VEC3;
    positions_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    positions_accessor.count = num_of_vertices;
    byte_length = 3 * num_of_vertices * sizeof(float);
    positions_accessor.bufferView = int(mesh_attributes_buffer_view_index);
    positions_accessor.byteOffset =
            model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
    model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
            byte_length;

    std::vector<unsigned char> mesh_attribute_buffer;
    for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
        const Eigen::Vector3d& vertex = mesh.vertices_[vidx];
        for (size_t i = 0; i < 3; ++i) {
            float_temp = (float)vertex(i);
            temp = (unsigned char*)&(float_temp);
            for (size_t j = 0; j < sizeof(float); ++j) {
                mesh_attribute_buffer.push_back(temp[j]);
            }
        }
    }

    Eigen::Vector3d min_bound = mesh.GetMinBound();
    positions_accessor.minValues.push_back(min_bound[0]);
    positions_accessor.minValues.push_back(min_bound[1]);
    positions_accessor.minValues.push_back(min_bound[2]);
    Eigen::Vector3d max_bound = mesh.GetMaxBound();
    positions_accessor.maxValues.push_back(max_bound[0]);
    positions_accessor.maxValues.push_back(max_bound[1]);
    positions_accessor.maxValues.push_back(max_bound[2]);
    model.accessors.push_back(positions_accessor);
    gltf_primitive.attributes.insert(std::make_pair(
            "POSITION", static_cast<int>(model.accessors.size()) - 1));

    write_vertex_normals = write_vertex_normals && mesh.HasVertexNormals();
    if (write_vertex_normals) {
        tinygltf::Accessor normals_accessor;
        normals_accessor.name = "buffer-0-accessor-normal-buffer-0-mesh-0";
        normals_accessor.type = TINYGLTF_TYPE_VEC3;
        normals_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        normals_accessor.count = mesh.vertices_.size();
        size_t byte_length = 3 * mesh.vertices_.size() * sizeof(float);
        normals_accessor.bufferView = int(mesh_attributes_buffer_view_index);
        normals_accessor.byteOffset =
                model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
        model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
                byte_length;

        for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
            const Eigen::Vector3d& normal = mesh.vertex_normals_[vidx];
            for (size_t i = 0; i < 3; ++i) {
                float_temp = (float)normal(i);
                temp = (unsigned char*)&(float_temp);
                for (size_t j = 0; j < sizeof(float); ++j) {
                    mesh_attribute_buffer.push_back(temp[j]);
                }
            }
        }

        model.accessors.push_back(normals_accessor);
        gltf_primitive.attributes.insert(std::make_pair(
                "NORMAL", static_cast<int>(model.accessors.size()) - 1));
    }

    write_vertex_colors = write_vertex_colors && mesh.HasVertexColors();
    if (write_vertex_colors) {
        tinygltf::Accessor colors_accessor;
        colors_accessor.name = "buffer-0-accessor-color-buffer-0-mesh-0";
        colors_accessor.type = TINYGLTF_TYPE_VEC3;
        colors_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        colors_accessor.count = mesh.vertices_.size();
        size_t byte_length = 3 * mesh.vertices_.size() * sizeof(float);
        colors_accessor.bufferView = int(mesh_attributes_buffer_view_index);
        colors_accessor.byteOffset =
                model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
        model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
                byte_length;

        for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
            const Eigen::Vector3d& color = mesh.vertex_colors_[vidx];
            for (size_t i = 0; i < 3; ++i) {
                float_temp = (float)color(i);
                temp = (unsigned char*)&(float_temp);
                for (size_t j = 0; j < sizeof(float); ++j) {
                    mesh_attribute_buffer.push_back(temp[j]);
                }
            }
        }

        model.accessors.push_back(colors_accessor);
        gltf_primitive.attributes.insert(std::make_pair(
                "COLOR_0", static_cast<int>(model.accessors.size()) - 1));
    }

    gltf_primitive.mode = TINYGLTF_MODE_TRIANGLES;
    gltf_mesh.primitives.push_back(gltf_primitive);
    model.meshes.push_back(gltf_mesh);

    model.bufferViews[0].byteOffset = 0;
    model.bufferViews[1].byteOffset = index_buffer.size();

    tinygltf::Buffer buffer;
    buffer.uri = filename.substr(0, filename.find_last_of(".")) + ".bin";
    buffer.data.resize(index_buffer.size() + mesh_attribute_buffer.size());
    memcpy(buffer.data.data(), index_buffer.data(), index_buffer.size());
    memcpy(buffer.data.data() + index_buffer.size(),
           mesh_attribute_buffer.data(), mesh_attribute_buffer.size());
    model.buffers.push_back(buffer);

    tinygltf::TinyGLTF loader;
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext == "glb") {
        if (!loader.WriteGltfSceneToFile(&model, filename, false, true, true,
                                         true)) {
            utility::LogWarning("Write GLTF failed.");
            return false;
        }
    } else {
        if (!loader.WriteGltfSceneToFile(&model, filename, false, true, true,
                                         false)) {
            utility::LogWarning("Write GLTF failed.");
            return false;
        }
    }

    return true;
}

}  // namespace io
}  // namespace open3d

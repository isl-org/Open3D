// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <tiny_gltf.h>

#include <numeric>
#include <vector>

#include <tcbspan/span.hpp>

#include "open3d/io/FileFormatIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/io/ModelIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/Model.h"

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

FileGeometry ReadFileGeometryTypeGLTF(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromGLTF(const std::string& filename,
                              geometry::TriangleMesh& mesh,
                              const ReadTriangleMeshOptions& params /*={}*/) {
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
            utility::LogWarning("Write GLB failed.");
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

template <typename T>
tcb::span<T, tcb::dynamic_extent> GetBufferSpan(
        tinygltf::Buffer& buff,
        const tinygltf::BufferView& view,
        const tinygltf::Accessor& accessor) {
    if (view.byteStride != 0 && view.byteStride != sizeof(T)) {
        utility::LogError("Cannot get a strided buffer span");
    }
    // Make sure the buffer can hold that much data
    buff.data.resize(view.byteOffset + view.byteLength);
    // Create a span with the correct type aliasing the underlying data
    return tcb::span<T, tcb::dynamic_extent>(
            reinterpret_cast<T*>(buff.data.data() +
                view.byteOffset + accessor.byteOffset),
            accessor.count);
}

tinygltf::Material ConvertMaterial(const visualization::rendering::MaterialRecord& mat_rec) {
    tinygltf::Material material;
    material.name = mat_rec.name;
    material.emissiveFactor = {{
            mat_rec.emissive_color(0),
            mat_rec.emissive_color(1),
            mat_rec.emissive_color(2),
    }};
    material.alphaMode = mat_rec.has_alpha ? "BLEND" : "OPAQUE";

    material.pbrMetallicRoughness.baseColorFactor = {{
            mat_rec.base_color(0),
            mat_rec.base_color(1),
            mat_rec.base_color(2),
            mat_rec.base_color(3),
    }};
    material.pbrMetallicRoughness.metallicFactor = mat_rec.base_metallic;
    material.pbrMetallicRoughness.roughnessFactor = mat_rec.base_roughness;

    if (mat_rec.thickness != 1.f || mat_rec.transmission != 1.f ||
        mat_rec.absorption_color != Eigen::Vector3f::Ones() ||
        mat_rec.absorption_distance != 1.f) {
        utility::LogWarning("Refractive materials are not supported when "
                "exporting to GLTF");
    }

    if (mat_rec.point_size != 3.f || mat_rec.line_width != 1.f) {
        utility::LogWarning("Line and Point materials are not supported "
                "when exporting to GLTF");
    }

    if (mat_rec.base_reflectance != 0.5f || mat_rec.reflectance_img) {
        utility::LogWarning(
                "Reflectance is not supported when exporting to GLTF");
    }

    if (mat_rec.base_clearcoat != 0.f ||
        mat_rec.base_clearcoat_roughness != 0.f ||
        mat_rec.clearcoat_img ||
        mat_rec.clearcoat_roughness_img ||
        mat_rec.ao_rough_metal_img) {
        utility::LogWarning(
                "Clearcoat is not supported when exporting to GLTF");
    }

    if (mat_rec.base_anisotropy != 0.f ||
        mat_rec.anisotropy_img) {
        utility::LogWarning(
                "Anisotropy is not supported when exporting to GLTF");
    }

    if (mat_rec.ao_rough_metal_img) {
        utility::LogWarning("Combined AO/Roughness is not supported "
                "when exporting to GLTF");
    }

    if (mat_rec.gradient ||
        mat_rec.scalar_min != 0.f ||
        mat_rec.scalar_max != 1.f) {
        utility::LogWarning("Gradient sampling is not supported "
                "when exporting to GLTF");
    }

    for (const auto& kvpair: mat_rec.generic_params) {
        utility::LogWarning("Skipping material property {}", kvpair.first);
    }

    for (const auto& kvpair: mat_rec.generic_imgs) {
        utility::LogWarning("Skipping material texture {}", kvpair.first);
    }
    return material;
}

void SerializeTexture(tinygltf::Model& model,
                      const geometry::Image& tex_img,
                      std::size_t primitive_idx,
                      const std::string& tex_name) {
    tinygltf::Image image;
    image.name = tex_img.GetName();
    if (image.name.empty()) {
        image.name = fmt::format("primitive-{}-{}", primitive_idx, tex_name);
    }
    image.mimeType = "image/jpeg";
    image.width = tex_img.width_;
    image.height = tex_img.height_;
    image.component = tex_img.num_of_channels_;
    image.bits = 8;
    image.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    // Copy the image data itself
    if (tex_img.bytes_per_channel_ == 1) {
        image.image = tex_img.data_;
    } else if (tex_img.bytes_per_channel_ == 2) {
        tcb::span<const std::uint16_t> input_span(
                reinterpret_cast<const std::uint16_t*>(tex_img.data_.data()),
                tex_img.data_.size() / 2);
        image.image.resize(input_span.size());
        tcb::span<std::uint8_t> output_span(
                image.image.data(), image.image.size());
        std::transform(input_span.begin(), input_span.end(),
                       output_span.begin(), [](const auto& val){
            return val;
        });
    } else {
        utility::LogError("Primitive {} cannot export {} image for GLTF",
                          primitive_idx, tex_name);
    }
    model.images.emplace_back(std::move(image));
    model.textures.emplace_back();
    model.textures.back().name = fmt::format(
            "primitive-{}-{}", primitive_idx, tex_name);
    model.textures.back().source =
            static_cast<int>(model.images.size()) - 1;
}

void WriteImagesToBuffers(tinygltf::Model& model) {
    for (tinygltf::Image& image: model.images) {
        model.buffers.emplace_back();
        tinygltf::Buffer& img_buff = model.buffers.back();
        img_buff.name = image.name + "-image-buffer";

        model.bufferViews.emplace_back();
        tinygltf::BufferView& img_buff_view = model.bufferViews.back();
        img_buff_view.name = image.name + "-image-buff-view";
        img_buff_view.buffer = static_cast<int>(model.buffers.size()) - 1;
        if (image.mimeType == "image/png") {
            if (!stbi_write_png_to_func(tinygltf::WriteToMemory_stbi,
                        &img_buff.data, image.width, image.height,
                        image.component, image.image.data(), 0)) {
                utility::LogError("Failed to serialize {}", image.name);
            }
        } else if (image.mimeType == "image/jpeg") {
            if (!stbi_write_jpg_to_func(tinygltf::WriteToMemory_stbi,
                        &img_buff.data, image.width, image.height,
                        image.component, image.image.data(), 100)) {
                utility::LogError("Failed to serialize {}", image.name);
            }
        } else if (image.mimeType == "image/bmp") {
            if (!stbi_write_bmp_to_func(tinygltf::WriteToMemory_stbi,
                        &img_buff.data, image.width, image.height,
                        image.component, image.image.data())) {
                utility::LogError("Failed to serialize {}", image.name);
            }
        } else {
            utility::LogError("Unsupported mime-type for image {}", image.name);
        }
        image.image.clear();
        image.bufferView = static_cast<int>(model.bufferViews.size()) - 1;
        image.as_is = true;
        img_buff_view.byteLength = img_buff.data.size();
    }
}

void ShrinkBuffersToFit(tinygltf::Model& model) {
    std::vector<std::size_t> sizes(model.buffers.size(), 0);
    for (const auto& view: model.bufferViews) {
        sizes[view.buffer] += view.byteLength;
    }
    for (std::size_t i = 0; i < model.buffers.size(); ++i) {
        model.buffers[i].data.resize(sizes[i]);
    }
}

void ConsolidateBuffers(tinygltf::Model& model) {
    ShrinkBuffersToFit(model);
    for (int i = 1; i < static_cast<int>(model.buffers.size()); ++i) {
        for (tinygltf::BufferView& view: model.bufferViews) {
            if (view.buffer != i) { continue; }
            view.byteOffset += model.buffers[0].data.size();
            view.buffer = 0;
        }
        model.buffers[0].data.insert(model.buffers[0].data.end(),
            model.buffers[i].data.begin(), model.buffers[i].data.end());
    }
    model.buffers.erase(std::next(model.buffers.begin()), model.buffers.end());
}

bool WriteTriangleMeshModelToGLTF(
        const std::string& filename,
        const visualization::rendering::TriangleMeshModel& mesh_model) {
    for (const auto& mesh_info: mesh_model.meshes_) {
        auto mat_it = std::minmax_element(
                mesh_info.mesh->triangle_material_ids_.begin(),
                mesh_info.mesh->triangle_material_ids_.end());
        if (mat_it.first != mat_it.second) {
            utility::LogWarning("Cannot export model because mesh {} has more "
                    "than one material", mesh_info.mesh_name);
            return false;
        }
    }
    std::string base_name =
            utility::filesystem::GetFileNameWithoutDirectory(filename);
    base_name = utility::filesystem::GetFileNameWithoutExtension(base_name);
    tinygltf::Model model;
    model.asset.generator = "Open3d";
    model.asset.version = "2.0";
    model.meshes.emplace_back();
    model.meshes.back().name = base_name;
    model.nodes.emplace_back();
    model.nodes.back().mesh = 0;
    model.scenes.emplace_back();
    model.scenes.back().nodes.emplace_back(0);
    model.defaultScene = 0;

    for (std::size_t i = 0; i < mesh_model.meshes_.size(); ++i) {
        geometry::TriangleMesh mesh;
        std::vector<Eigen::Vector2d> vertex_uvs;
        std::tie(mesh, vertex_uvs) =
                detail::MeshWithPerVertexUVs(*mesh_model.meshes_[i].mesh);
        if (!mesh.HasTriangles()) {
            utility::LogWarning("Invalid Mesh {}:{} has no triangles and will "
                    "be skipped", i, mesh_model.meshes_[i].mesh_name);
            continue;
        }

        model.meshes.back().primitives.emplace_back();
        tinygltf::Primitive& primitive = model.meshes.back().primitives.back();
        primitive.mode = TINYGLTF_MODE_TRIANGLES;

        model.buffers.emplace_back();
        tinygltf::Buffer& mesh_buffer = model.buffers.back();
        mesh_buffer.name = fmt::format(
                "primitive-{}-geometry-buffer", i);

        // Store triangle indices
        bool save_indices_as_uint16 = mesh.vertices_.size() <=
            std::numeric_limits<std::uint16_t>::max();

        model.bufferViews.emplace_back();
        tinygltf::BufferView& indices_view = model.bufferViews.back();
        indices_view.name = fmt::format("primitive-{}-index-buffview", i);
        indices_view.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
        indices_view.buffer = static_cast<int>(model.buffers.size()) - 1;

        tinygltf::Accessor indices_accaessor;
        indices_accaessor.bufferView =
                static_cast<int>(model.bufferViews.size()) - 1;
        indices_accaessor.name = fmt::format("primitive-{}-indices", i);
        indices_accaessor.type = TINYGLTF_TYPE_SCALAR;
        indices_accaessor.componentType = save_indices_as_uint16 ?
                TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT :
                TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        indices_accaessor.count = 3 * mesh.triangles_.size();
        indices_accaessor.byteOffset = 0;
        indices_accaessor.minValues = {0};
        indices_accaessor.maxValues =
                {static_cast<double>(mesh.vertices_.size()) - 1};

        if (save_indices_as_uint16) {
            indices_view.byteLength = indices_accaessor.count *
                                      sizeof(std::uint16_t);
            using OutType = std::array<std::uint16_t, 3>;
            auto span = GetBufferSpan<OutType>(
                    mesh_buffer, indices_view, indices_accaessor);
            std::transform(mesh.triangles_.begin(), mesh.triangles_.end(),
                    span.begin(), [](const Eigen::Vector3i& tri){
                const auto tdat = tri.template cast<OutType::value_type>();
                return OutType{tdat(0), tdat(1), tdat(2)};
            });
        } else {
            if (mesh.vertices_.size() >
                std::numeric_limits<std::uint32_t>::max()) {
                utility::LogError("Number of vertices {} is unsupported "
                        "by this writer", mesh.vertices_.size());
            }
            indices_view.byteLength = indices_accaessor.count *
                                      sizeof(std::uint32_t);
            using OutType = std::array<std::uint32_t, 3>;
            auto span = GetBufferSpan<OutType>(
                    mesh_buffer, indices_view, indices_accaessor);
            std::transform(mesh.triangles_.begin(), mesh.triangles_.end(),
                    span.begin(), [](const Eigen::Vector3i& tri){
                const auto tdat = tri.template cast<OutType::value_type>();
                return OutType{tdat(0), tdat(1), tdat(2)};
            });
        }
        // Add the indices to the primitive
        model.accessors.emplace_back(std::move(indices_accaessor));
        primitive.indices = static_cast<int>(model.accessors.size()) - 1;

        model.bufferViews.emplace_back();
        tinygltf::BufferView& data_view_strided = model.bufferViews.back();
        data_view_strided.name = fmt::format(
                "primitive-{}-vertdata-buffview", i);
        data_view_strided.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        data_view_strided.byteOffset = indices_view.byteLength;
        data_view_strided.byteStride = 12;
        data_view_strided.buffer = static_cast<int>(model.buffers.size()) - 1;
        int data_view_idx = static_cast<int>(model.bufferViews.size()) - 1;

        tinygltf::Accessor vertex_accessor;
        vertex_accessor.name = fmt::format("primitive-{}-vertices", i);
        vertex_accessor.type = TINYGLTF_TYPE_VEC3;
        vertex_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        vertex_accessor.count = mesh.vertices_.size();
        vertex_accessor.bufferView = data_view_idx;
        vertex_accessor.minValues = std::vector<double>(
                3, std::numeric_limits<double>::max());
        vertex_accessor.maxValues = std::vector<double>(
                3, std::numeric_limits<double>::lowest());
        vertex_accessor.byteOffset = data_view_strided.byteLength;
        data_view_strided.byteLength += 3 * sizeof(float) * vertex_accessor.count;
        {
            using OutType = std::array<float, 3>;
            auto span = GetBufferSpan<OutType>(
                    mesh_buffer, data_view_strided, vertex_accessor);
            std::transform(mesh.vertices_.begin(), mesh.vertices_.end(),
                    span.begin(), [&vertex_accessor](const auto& vert){
                auto vout = vert.template cast<OutType::value_type>();
                // Change coordinates system
                return OutType{vout(0), vout(2), -vout(1)};
            });
            std::for_each(span.begin(), span.end(),
                          [&vertex_accessor](const auto& arr){
                for (int i = 0; i < 3; ++i) {
                    vertex_accessor.minValues[i] = std::min<double>(
                            arr[i], vertex_accessor.minValues[i]);
                    vertex_accessor.maxValues[i] = std::max<double>(
                            arr[i], vertex_accessor.maxValues[i]);
                }
            });
        }
        // Register the positions accessor to the primitive
        model.accessors.emplace_back(std::move(vertex_accessor));
        primitive.attributes["POSITION"] =
                static_cast<int>(model.accessors.size()) - 1;

        if (mesh.HasVertexNormals()) {
            tinygltf::Accessor normals_accessor;
            normals_accessor.name = fmt::format("primitive-{}-normals", i);
            normals_accessor.type = TINYGLTF_TYPE_VEC3;
            normals_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            normals_accessor.count = mesh.vertex_normals_.size();
            normals_accessor.bufferView = data_view_idx;
            normals_accessor.byteOffset = data_view_strided.byteLength;
            data_view_strided.byteLength += 3 * sizeof(float) * normals_accessor.count;
            {
                using OutType = std::array<float, 3>;
                auto span = GetBufferSpan<OutType>(
                        mesh_buffer, data_view_strided, normals_accessor);
                std::transform(mesh.vertex_normals_.begin(),
                               mesh.vertex_normals_.end(),
                               span.begin(), [](const auto& norm){
                    auto nout = norm.template cast<OutType::value_type>();
                    return OutType{nout(0), nout(1), nout(2)};
                });
            }
            // Register the normals accessor to the primitive
            model.accessors.emplace_back(std::move(normals_accessor));
            primitive.attributes["NORMAL"] =
                    static_cast<int>(model.accessors.size()) - 1;
        }
        
        if (mesh.HasVertexColors()) {
            tinygltf::Accessor colors_accessor;
            colors_accessor.name = fmt::format("primitive-{}-colors", i);
            colors_accessor.type = TINYGLTF_TYPE_VEC3;
            colors_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            colors_accessor.count = mesh.vertex_colors_.size();
            colors_accessor.bufferView = data_view_idx;
            colors_accessor.byteOffset = data_view_strided.byteLength;
            data_view_strided.byteLength += 3 * sizeof(float) * colors_accessor.count;
            {
                using OutType = std::array<float, 3>;
                auto span = GetBufferSpan<OutType>(
                        mesh_buffer, data_view_strided, colors_accessor);
                std::transform(mesh.vertex_colors_.begin(),
                               mesh.vertex_colors_.end(),
                               span.begin(), [](const auto& color){
                    auto cout = color.template cast<OutType::value_type>();
                    return OutType{cout(0), cout(1), cout(2)};
                });
            }
            // Register the colors accessor to the primitive
            model.accessors.emplace_back(std::move(colors_accessor));
            primitive.attributes["COLOR_0"] =
                    static_cast<int>(model.accessors.size()) - 1;
        }

        if (mesh.HasTriangleNormals()) {
            utility::LogWarning("Mesh {}:{} has per-triangle normals which "
                    "are not supported and will be skipped",
                    i, mesh_model.meshes_[i].mesh_name);
        }

        if (mesh.HasAdjacencyList()) {
            utility::LogWarning("Mesh {}:{} has an adjacency list which "
                    "is not supported and will be skipped",
                    i, mesh_model.meshes_[i].mesh_name);
        }

        if (mesh.HasTriangleUvs()) {
            model.bufferViews.emplace_back();
            tinygltf::BufferView& uv_buffer_view = model.bufferViews.back();
            uv_buffer_view.name = fmt::format(
                    "primitive-{}-uvdata-buffview", i);
            uv_buffer_view.target = TINYGLTF_TARGET_ARRAY_BUFFER;
            uv_buffer_view.byteOffset =
                    data_view_strided.byteOffset + data_view_strided.byteLength;
            uv_buffer_view.byteStride = 8;
            uv_buffer_view.buffer = static_cast<int>(model.buffers.size()) - 1;

            tinygltf::Accessor uv_accessor;
            uv_accessor.name = fmt::format("primitive-{}-uvs", i);
            uv_accessor.type = TINYGLTF_TYPE_VEC2;
            uv_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            uv_accessor.count = vertex_uvs.size();
            uv_accessor.bufferView =
                    static_cast<int>(model.bufferViews.size()) - 1;
            uv_accessor.byteOffset = uv_buffer_view.byteLength;
            uv_buffer_view.byteLength += 2 * sizeof(float) * uv_accessor.count;
            {
                using OutType = std::array<float, 2>;
                auto span = GetBufferSpan<OutType>(
                        mesh_buffer, uv_buffer_view, uv_accessor);
                std::transform(vertex_uvs.begin(), vertex_uvs.end(),
                               span.begin(), [](const auto& uvs){
                    auto uvout = uvs.template cast<OutType::value_type>();
                    return OutType{uvout(0), 1.f - uvout(1)};
                });
            }
            // Register the uv accessor to the primitive
            model.accessors.emplace_back(std::move(uv_accessor));
            primitive.attributes["TEXCOORD_0"] =
                    static_cast<int>(model.accessors.size()) - 1;
        }

        // Write the material definition
        const auto& mat_rec = mesh_model.materials_[i];
        tinygltf::Material material = ConvertMaterial(mat_rec);

        // Write the textures
        if (mat_rec.albedo_img && mat_rec.albedo_img->HasData()) {
            SerializeTexture(model, *mat_rec.albedo_img, i, "albedo");
            material.pbrMetallicRoughness.baseColorTexture.index =
                    static_cast<int>(model.textures.size()) - 1;
        }
        if (mat_rec.roughness_img && mat_rec.roughness_img->HasData()) {
            SerializeTexture(model, *mat_rec.roughness_img, i, "roughness");
            material.pbrMetallicRoughness.metallicRoughnessTexture.index =
                    static_cast<int>(model.textures.size()) - 1;
        }
        if (mat_rec.normal_img && mat_rec.normal_img->HasData()) {
            SerializeTexture(model, *mat_rec.normal_img, i, "normals");
            material.normalTexture.index =
                    static_cast<int>(model.textures.size()) - 1;
        }
        if (mat_rec.ao_img && mat_rec.ao_img->HasData()) {
            SerializeTexture(model, *mat_rec.ao_img, i, "ao");
            material.occlusionTexture.index =
                    static_cast<int>(model.textures.size()) - 1;
        }
        model.materials.emplace_back(std::move(material));
        primitive.material = static_cast<int>(model.materials.size()) - 1;
    }

    tinygltf::TinyGLTF loader;
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    ShrinkBuffersToFit(model);
    if (filename_ext == "glb") {
        WriteImagesToBuffers(model);
        ShrinkBuffersToFit(model);
        ConsolidateBuffers(model);
        if (!loader.WriteGltfSceneToFile(
                    &model, filename, true, true, true, true)) {
            utility::LogWarning("Write GLB failed.");
            return false;
        }
    } else {
        if (!loader.WriteGltfSceneToFile(
                    &model, filename, false, false, true, false)) {
            utility::LogWarning("Write GLTF failed.");
            return false;
        }
    }
    return true;
}

}  // namespace io
}  // namespace open3d

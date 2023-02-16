// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/geometry/VoxelBlockGrid.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace t {
namespace geometry {

static std::pair<core::Tensor, core::Tensor> BufferRadiusNeighbors(
        std::shared_ptr<core::HashMap> &hashmap,
        const core::Tensor &active_buf_indices) {
    // Fixed radius search for spatially hashed voxel blocks.
    // A generalization will be implementing dense/sparse fixed radius
    // search with coordinates as hashmap keys.
    core::Tensor key_buffer_int3_tensor = hashmap->GetKeyTensor();

    core::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
            {active_buf_indices.To(core::Int64)});
    int64_t n = active_keys.GetShape()[0];

    // Fill in radius nearest neighbors.
    core::Tensor keys_nb({27, n, 3}, core::Int32, hashmap->GetDevice());
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        core::Tensor dt =
                core::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1}, {1, 3},
                             core::Int32, hashmap->GetDevice());
        keys_nb[nb] = active_keys + dt;
    }
    keys_nb = keys_nb.View({27 * n, 3});

    core::Tensor buf_indices_nb, masks_nb;
    hashmap->Find(keys_nb, buf_indices_nb, masks_nb);
    return std::make_pair(buf_indices_nb.View({27, n, 1}),
                          masks_nb.View({27, n, 1}));
}

static TensorMap ConstructTensorMap(
        const core::HashMap &block_hashmap,
        std::unordered_map<std::string, int> name_attr_map) {
    TensorMap tensor_map("tsdf");
    for (auto &v : name_attr_map) {
        std::string name = v.first;
        int buf_idx = v.second;
        tensor_map[name] = block_hashmap.GetValueTensor(buf_idx);
    }
    return tensor_map;
}

VoxelBlockGrid::VoxelBlockGrid(
        const std::vector<std::string> &attr_names,
        const std::vector<core::Dtype> &attr_dtypes,
        const std::vector<core::SizeVector> &attr_channels,
        float voxel_size,
        int64_t block_resolution,
        int64_t block_count,
        const core::Device &device,
        const core::HashBackendType &backend)
    : voxel_size_(voxel_size), block_resolution_(block_resolution) {
    // Sanity check
    if (voxel_size <= 0) {
        utility::LogError("voxel size must be positive, but got {}",
                          voxel_size);
    }
    if (block_resolution <= 0) {
        utility::LogError("block resolution must be positive, but got {}",
                          block_resolution);
    }

    // Check property lengths
    size_t n_attrs = attr_names.size();
    if (attr_dtypes.size() != n_attrs) {
        utility::LogError(
                "Number of attribute dtypes ({}) mismatch with names ({}).",
                attr_dtypes.size(), n_attrs);
    }
    if (attr_channels.size() != n_attrs) {
        utility::LogError(
                "Number of attribute channels ({}) mismatch with names ({}).",
                attr_channels.size(), n_attrs);
    }

    // Specify block element shapes and attribute names.
    std::vector<core::SizeVector> attr_element_shapes;
    core::SizeVector block_shape{block_resolution, block_resolution,
                                 block_resolution};
    for (size_t i = 0; i < n_attrs; ++i) {
        // Construct element shapes.
        core::SizeVector attr_channel = attr_channels[i];
        core::SizeVector block_shape_copy = block_shape;
        block_shape_copy.insert(block_shape_copy.end(), attr_channel.begin(),
                                attr_channel.end());
        attr_element_shapes.emplace_back(block_shape_copy);

        // Used for easier accessing via attribute names.
        name_attr_map_[attr_names[i]] = i;
    }

    block_hashmap_ = std::make_shared<core::HashMap>(
            block_count, core::Int32, core::SizeVector{3}, attr_dtypes,
            attr_element_shapes, device, backend);
}

core::Tensor VoxelBlockGrid::GetAttribute(const std::string &attr_name) const {
    AssertInitialized();
    if (name_attr_map_.count(attr_name) == 0) {
        utility::LogWarning("Attribute {} not found, return empty tensor.",
                            attr_name);
        return core::Tensor();
    }
    int buffer_idx = name_attr_map_.at(attr_name);
    return block_hashmap_->GetValueTensor(buffer_idx);
}

core::Tensor VoxelBlockGrid::GetVoxelCoordinates(
        const core::Tensor &voxel_indices) const {
    AssertInitialized();
    core::Tensor key_tensor = block_hashmap_->GetKeyTensor();

    core::Tensor voxel_coords =
            key_tensor.IndexGet({voxel_indices[0]}).T().To(core::Int64) *
            block_resolution_;
    voxel_coords[0] += voxel_indices[1];
    voxel_coords[1] += voxel_indices[2];
    voxel_coords[2] += voxel_indices[3];

    return voxel_coords;
}

core::Tensor VoxelBlockGrid::GetVoxelIndices(
        const core::Tensor &buf_indices) const {
    AssertInitialized();
    core::Device device = block_hashmap_->GetDevice();

    int64_t n_blocks = buf_indices.GetLength();

    int64_t resolution = block_resolution_;
    int64_t resolution2 = resolution * resolution;
    int64_t resolution3 = resolution2 * resolution;

    // Non-kernel version.
    /// TODO: Check if kernel version is necessary.
    core::Tensor linear_coordinates = core::Tensor::Arange(
            0, n_blocks * resolution3, 1, core::Int64, device);

    core::Tensor block_idx = linear_coordinates / resolution3;
    core::Tensor remainder = linear_coordinates - block_idx * resolution3;

    /// operator % is not supported now
    core::Tensor voxel_z = remainder / resolution2;
    remainder = remainder - voxel_z * resolution2;
    core::Tensor voxel_y = remainder / resolution;
    core::Tensor voxel_x = remainder - voxel_y * resolution;

    core::Tensor voxel_indices = core::Tensor({4, n_blocks * resolution3},
                                              core::Dtype::Int64, device);
    voxel_indices[0] = buf_indices.IndexGet({block_idx}).To(core::Dtype::Int64);
    voxel_indices[1] = voxel_x;
    voxel_indices[2] = voxel_y;
    voxel_indices[3] = voxel_z;

    return voxel_indices;
}

core::Tensor VoxelBlockGrid::GetVoxelIndices() const {
    AssertInitialized();
    return GetVoxelIndices(block_hashmap_->GetActiveIndices());
}

std::pair<core::Tensor, core::Tensor>
VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices() {
    AssertInitialized();
    return GetVoxelCoordinatesAndFlattenedIndices(
            block_hashmap_->GetActiveIndices());
}

std::pair<core::Tensor, core::Tensor>
VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices(
        const core::Tensor &buf_indices) {
    AssertInitialized();
    // (N x resolution^3, 3) Float32; (N x resolution^3, 1) Int64
    int64_t n = buf_indices.GetLength();

    int64_t resolution3 =
            block_resolution_ * block_resolution_ * block_resolution_;

    core::Device device = block_hashmap_->GetDevice();
    core::Tensor voxel_coords({n * resolution3, 3}, core::Float32, device);
    core::Tensor flattened_indices({n * resolution3}, core::Int64, device);

    kernel::voxel_grid::GetVoxelCoordinatesAndFlattenedIndices(
            buf_indices, block_hashmap_->GetKeyTensor(), voxel_coords,
            flattened_indices, block_resolution_, voxel_size_);
    return std::make_pair(voxel_coords, flattened_indices);
}

core::Tensor VoxelBlockGrid::GetUniqueBlockCoordinates(
        const Image &depth,
        const core::Tensor &intrinsic,
        const core::Tensor &extrinsic,
        float depth_scale,
        float depth_max,
        float trunc_voxel_multiplier) {
    AssertInitialized();
    CheckDepthTensor(depth.AsTensor());
    CheckIntrinsicTensor(intrinsic);
    CheckExtrinsicTensor(extrinsic);

    const int64_t down_factor = 4;
    const int64_t est_sample_multiplier = 4;
    if (frustum_hashmap_ == nullptr) {
        int64_t capacity = (depth.GetCols() / down_factor) *
                           (depth.GetRows() / down_factor) *
                           est_sample_multiplier;
        frustum_hashmap_ = std::make_shared<core::HashMap>(
                capacity, core::Int32, core::SizeVector{3}, core::Int32,
                core::SizeVector{1}, block_hashmap_->GetDevice());
    } else {
        frustum_hashmap_->Clear();
    }

    core::Tensor block_coords;
    kernel::voxel_grid::DepthTouch(frustum_hashmap_, depth.AsTensor(),
                                   intrinsic, extrinsic, block_coords,
                                   block_resolution_, voxel_size_,
                                   voxel_size_ * trunc_voxel_multiplier,
                                   depth_scale, depth_max, down_factor);

    return block_coords;
}

core::Tensor VoxelBlockGrid::GetUniqueBlockCoordinates(
        const PointCloud &pcd, float trunc_voxel_multiplier) {
    AssertInitialized();
    core::Tensor positions = pcd.GetPointPositions();

    const int64_t est_neighbor_multiplier = 8;
    if (frustum_hashmap_ == nullptr) {
        int64_t capacity = positions.GetLength() * est_neighbor_multiplier;
        frustum_hashmap_ = std::make_shared<core::HashMap>(
                capacity, core::Int32, core::SizeVector{3}, core::Int32,
                core::SizeVector{1}, block_hashmap_->GetDevice());
    } else {
        frustum_hashmap_->Clear();
    }

    core::Tensor block_coords;
    kernel::voxel_grid::PointCloudTouch(
            frustum_hashmap_, positions, block_coords, block_resolution_,
            voxel_size_, voxel_size_ * trunc_voxel_multiplier);
    return block_coords;
}

void VoxelBlockGrid::Integrate(const core::Tensor &block_coords,
                               const Image &depth,
                               const core::Tensor &intrinsic,
                               const core::Tensor &extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
    Integrate(block_coords, depth, Image(), intrinsic, intrinsic, extrinsic,
              depth_scale, depth_max, trunc_voxel_multiplier);
}

void VoxelBlockGrid::Integrate(const core::Tensor &block_coords,
                               const Image &depth,
                               const Image &color,
                               const core::Tensor &intrinsic,
                               const core::Tensor &extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
    Integrate(block_coords, depth, color, intrinsic, intrinsic, extrinsic,
              depth_scale, depth_max, trunc_voxel_multiplier);
}

void VoxelBlockGrid::Integrate(const core::Tensor &block_coords,
                               const Image &depth,
                               const Image &color,
                               const core::Tensor &depth_intrinsic,
                               const core::Tensor &color_intrinsic,
                               const core::Tensor &extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
    AssertInitialized();
    bool integrate_color = color.AsTensor().NumElements() > 0;

    CheckBlockCoorinates(block_coords);
    CheckDepthTensor(depth.AsTensor());
    if (integrate_color) {
        CheckColorTensor(color.AsTensor());
    }
    CheckIntrinsicTensor(depth_intrinsic);
    CheckIntrinsicTensor(color_intrinsic);
    CheckExtrinsicTensor(extrinsic);

    core::Tensor buf_indices, masks;
    block_hashmap_->Activate(block_coords, buf_indices, masks);
    block_hashmap_->Find(block_coords, buf_indices, masks);

    core::Tensor block_keys = block_hashmap_->GetKeyTensor();
    TensorMap block_value_map =
            ConstructTensorMap(*block_hashmap_, name_attr_map_);

    kernel::voxel_grid::Integrate(
            depth.AsTensor(), color.AsTensor(), buf_indices, block_keys,
            block_value_map, depth_intrinsic, color_intrinsic, extrinsic,
            block_resolution_, voxel_size_,
            voxel_size_ * trunc_voxel_multiplier, depth_scale, depth_max);
}

TensorMap VoxelBlockGrid::RayCast(const core::Tensor &block_coords,
                                  const core::Tensor &intrinsic,
                                  const core::Tensor &extrinsic,
                                  int width,
                                  int height,
                                  const std::vector<std::string> attrs,
                                  float depth_scale,
                                  float depth_min,
                                  float depth_max,
                                  float weight_threshold,
                                  float trunc_voxel_multiplier,
                                  int range_map_down_factor) {
    AssertInitialized();
    CheckBlockCoorinates(block_coords);
    CheckIntrinsicTensor(intrinsic);
    CheckExtrinsicTensor(extrinsic);

    // Extrinsic: world to camera -> pose: camera to world
    core::Device device = block_hashmap_->GetDevice();

    core::Tensor range_minmax_map;
    kernel::voxel_grid::EstimateRange(
            block_coords, range_minmax_map, intrinsic, extrinsic, height, width,
            range_map_down_factor, block_resolution_, voxel_size_, depth_min,
            depth_max, fragment_buffer_);

    static const std::unordered_map<std::string, int> kAttrChannelMap = {
            // Conventional rendering
            {"vertex", 3},
            {"normal", 3},
            {"depth", 1},
            {"color", 3},
            // Diff rendering
            // Each pixel corresponds to info at 8 neighbor grid points
            {"index", 8},
            {"mask", 8},
            {"interp_ratio", 8},
            {"interp_ratio_dx", 8},
            {"interp_ratio_dy", 8},
            {"interp_ratio_dz", 8}};

    auto get_dtype = [&](const std::string &attr_name) -> core::Dtype {
        if (attr_name == "mask") {
            return core::Dtype::Bool;
        } else if (attr_name == "index") {
            return core::Dtype::Int64;
        } else {
            return core::Dtype::Float32;
        }
    };

    TensorMap renderings_map("range");
    renderings_map["range"] = range_minmax_map;
    for (const auto &attr : attrs) {
        if (kAttrChannelMap.count(attr) == 0) {
            utility::LogError(
                    "Unsupported attribute {}, please implement customized ray "
                    "casting.");
        }
        int channel = kAttrChannelMap.at(attr);
        core::Dtype dtype = get_dtype(attr);
        renderings_map[attr] =
                core::Tensor({height, width, channel}, dtype, device);
    }

    TensorMap block_value_map =
            ConstructTensorMap(*block_hashmap_, name_attr_map_);
    kernel::voxel_grid::RayCast(
            block_hashmap_, block_value_map, range_minmax_map, renderings_map,
            intrinsic, extrinsic, height, width, block_resolution_, voxel_size_,
            depth_scale, depth_min, depth_max, weight_threshold,
            trunc_voxel_multiplier, range_map_down_factor);

    return renderings_map;
}

PointCloud VoxelBlockGrid::ExtractPointCloud(float weight_threshold,
                                             int estimated_point_number) {
    AssertInitialized();
    core::Tensor active_buf_indices;
    block_hashmap_->GetActiveIndices(active_buf_indices);

    core::Tensor active_nb_buf_indices, active_nb_masks;
    std::tie(active_nb_buf_indices, active_nb_masks) =
            BufferRadiusNeighbors(block_hashmap_, active_buf_indices);

    // Extract points around zero-crossings.
    core::Tensor points, normals, colors;

    core::Tensor block_keys = block_hashmap_->GetKeyTensor();
    TensorMap block_value_map =
            ConstructTensorMap(*block_hashmap_, name_attr_map_);
    kernel::voxel_grid::ExtractPointCloud(
            active_buf_indices, active_nb_buf_indices, active_nb_masks,
            block_keys, block_value_map, points, normals, colors,
            block_resolution_, voxel_size_, weight_threshold,
            estimated_point_number);

    auto pcd = PointCloud(points.Slice(0, 0, estimated_point_number));
    pcd.SetPointNormals(normals.Slice(0, 0, estimated_point_number));

    if (colors.GetLength() == normals.GetLength()) {
        pcd.SetPointColors(colors.Slice(0, 0, estimated_point_number));
    }

    return pcd;
}

TriangleMesh VoxelBlockGrid::ExtractTriangleMesh(float weight_threshold,
                                                 int estimated_vertex_number) {
    AssertInitialized();
    core::Tensor active_buf_indices_i32 = block_hashmap_->GetActiveIndices();
    core::Tensor active_nb_buf_indices, active_nb_masks;
    std::tie(active_nb_buf_indices, active_nb_masks) =
            BufferRadiusNeighbors(block_hashmap_, active_buf_indices_i32);

    core::Device device = block_hashmap_->GetDevice();
    // Map active indices to [0, num_blocks] to be allocated for surface mesh.
    int64_t num_blocks = block_hashmap_->Size();
    core::Tensor inverse_index_map({block_hashmap_->GetCapacity()}, core::Int32,
                                   device);
    core::Tensor iota_map =
            core::Tensor::Arange(0, num_blocks, 1, core::Int32, device);
    inverse_index_map.IndexSet({active_buf_indices_i32.To(core::Int64)},
                               iota_map);

    core::Tensor vertices, triangles, vertex_normals, vertex_colors;

    core::Tensor block_keys = block_hashmap_->GetKeyTensor();
    TensorMap block_value_map =
            ConstructTensorMap(*block_hashmap_, name_attr_map_);
    kernel::voxel_grid::ExtractTriangleMesh(
            active_buf_indices_i32, inverse_index_map, active_nb_buf_indices,
            active_nb_masks, block_keys, block_value_map, vertices, triangles,
            vertex_normals, vertex_colors, block_resolution_, voxel_size_,
            weight_threshold, estimated_vertex_number);

    TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexNormals(vertex_normals);
    if (vertex_colors.GetLength() == vertices.GetLength()) {
        mesh.SetVertexColors(vertex_colors);
    }

    return mesh;
}

void VoxelBlockGrid::Save(const std::string &file_name) const {
    AssertInitialized();
    // TODO(wei): provide 'GetActiveKeyValues' functionality.
    core::Tensor keys = block_hashmap_->GetKeyTensor();
    std::vector<core::Tensor> values = block_hashmap_->GetValueTensors();

    core::Device host("CPU:0");

    core::Tensor active_buf_indices_i32 = block_hashmap_->GetActiveIndices();
    core::Tensor active_indices = active_buf_indices_i32.To(core::Int64);

    std::unordered_map<std::string, core::Tensor> output;

    // Save name attributes
    output.emplace("voxel_size", core::Tensor(std::vector<float>{voxel_size_},
                                              {1}, core::Float32, host));
    output.emplace("block_resolution",
                   core::Tensor(std::vector<int64_t>{block_resolution_}, {1},
                                core::Int64, host));
    // Placeholder
    output.emplace(block_hashmap_->GetDevice().ToString(),
                   core::Tensor::Zeros({}, core::Dtype::UInt8, host));

    for (auto &it : name_attr_map_) {
        // Workaround, as we don't support char tensors now.
        output.emplace(fmt::format("attr_name_{}", it.first),
                       core::Tensor(std::vector<int>{it.second}, {1},
                                    core::Int32, host));
    }

    // Save keys
    core::Tensor active_keys = keys.IndexGet({active_indices}).To(host);
    output.emplace("key", active_keys);

    // Save SoA values and name attributes
    for (auto &it : name_attr_map_) {
        int value_id = it.second;
        core::Tensor active_value_i =
                values[value_id].IndexGet({active_indices}).To(host);
        output.emplace(fmt::format("value_{:03d}", value_id), active_value_i);
    }

    std::string ext =
            utility::filesystem::GetFileExtensionInLowerCase(file_name);
    if (ext != "npz") {
        utility::LogWarning(
                "File name for a voxel grid should be with the extension "
                ".npz. Saving to {}.npz",
                file_name);
        t::io::WriteNpz(file_name + ".npz", output);
    } else {
        t::io::WriteNpz(file_name, output);
    }
}

VoxelBlockGrid VoxelBlockGrid::To(const core::Device &device, bool copy) const {
    if (!copy && block_hashmap_->GetDevice() == device) {
        return *this;
    }

    auto device_hashmap =
            std::make_shared<core::HashMap>(this->block_hashmap_->To(device));
    return VoxelBlockGrid(voxel_size_, block_resolution_, device_hashmap,
                          name_attr_map_);
}

VoxelBlockGrid VoxelBlockGrid::Load(const std::string &file_name) {
    std::unordered_map<std::string, core::Tensor> tensor_map =
            t::io::ReadNpz(file_name);

    std::string prefix = "attr_name_";
    std::unordered_map<int, std::string> inv_attr_map;

    std::string kCPU = "CPU";
    std::string kCUDA = "CUDA";

    std::string device_str = "CPU:0";
    for (auto &it : tensor_map) {
        if (!it.first.compare(0, prefix.size(), prefix)) {
            int value_id = it.second[0].Item<int>();
            inv_attr_map.emplace(value_id, it.first.substr(prefix.size()));
        }
        if (!it.first.compare(0, kCPU.size(), kCPU) ||
            !it.first.compare(0, kCUDA.size(), kCUDA)) {
            device_str = it.first;
        }
    }
    if (inv_attr_map.size() == 0) {
        utility::LogError(
                "Attribute names not found, not a valid file for voxel block "
                "grids.");
    }

    core::Device device(device_str);

    std::vector<std::string> attr_names(inv_attr_map.size());

    std::vector<core::Tensor> soa_value_tensor(inv_attr_map.size());
    std::vector<core::Dtype> attr_dtypes(inv_attr_map.size());
    std::vector<core::SizeVector> attr_channels(inv_attr_map.size());

    // Not an ideal way to use an unordered map. Assume all the indices are
    // stored.
    for (auto &v : inv_attr_map) {
        int value_id = v.first;
        attr_names[value_id] = v.second;

        core::Tensor value_i =
                tensor_map.at(fmt::format("value_{:03d}", value_id));

        soa_value_tensor[value_id] = value_i.To(device);
        attr_dtypes[value_id] = value_i.GetDtype();

        core::SizeVector value_i_shape = value_i.GetShape();
        // capacity, res, res, res
        value_i_shape.erase(value_i_shape.begin(), value_i_shape.begin() + 4);
        attr_channels[value_id] = value_i_shape;
    }

    core::Tensor keys = tensor_map.at("key").To(device);
    float voxel_size = tensor_map.at("voxel_size")[0].Item<float>();
    int block_resolution = tensor_map.at("block_resolution")[0].Item<int64_t>();

    VoxelBlockGrid vbg(attr_names, attr_dtypes, attr_channels, voxel_size,
                       block_resolution, keys.GetLength(), device);
    auto block_hashmap = vbg.GetHashMap();
    block_hashmap.Insert(keys, soa_value_tensor);
    return vbg;
}

void VoxelBlockGrid::AssertInitialized() const {
    if (block_hashmap_ == nullptr) {
        utility::LogError("VoxelBlockGrid not initialized.");
    }
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d

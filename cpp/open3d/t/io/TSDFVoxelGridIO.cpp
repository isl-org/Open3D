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

#include "open3d/t/io/TSDFVoxelGridIO.h"

#include <json/json.h>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/t/io/HashmapIO.h"
#include "open3d/utility/FileSystem.h"
namespace open3d {
namespace t {
namespace io {

class TSDFVoxelGridMetadata : public utility::IJsonConvertible {
public:
    TSDFVoxelGridMetadata() = default;
    TSDFVoxelGridMetadata(const geometry::TSDFVoxelGrid &tsdf_voxelgrid) {
        voxel_size_ = tsdf_voxelgrid.voxel_size_;
        sdf_trunc_ = tsdf_voxelgrid.sdf_trunc_;

        block_resolution_ = tsdf_voxelgrid.block_resolution_;
        block_count_ = tsdf_voxelgrid.block_count_;
        device_ = tsdf_voxelgrid.device_.ToString();

        attr_dtype_map_ = tsdf_voxelgrid.attr_dtype_map_;
    }

    bool ConvertToJsonValue(Json::Value &value) const override {
        value["voxel_size"] = voxel_size_;
        value["sdf_trunc"] = sdf_trunc_;

        value["block_resolution"] = block_resolution_;
        value["block_count"] = block_count_;
        value["device"] = device_;
        value["hashmap_filename"] = hashmap_filename_;

        Json::Value attr_dtype_map;
        for (auto it : attr_dtype_map_) {
            attr_dtype_map[it.first] = it.second.ToString();
        }

        value["attr_dtype_map"] = attr_dtype_map;
        return true;
    }

    bool ConvertFromJsonValue(const Json::Value &value) override {
        voxel_size_ = value["voxel_size"].asFloat();
        sdf_trunc_ = value["sdf_trunc"].asFloat();

        block_resolution_ = value["block_resolution"].asInt64();
        block_count_ = value["block_count"].asInt64();
        device_ = value["device"].asString();
        hashmap_filename_ = value["hashmap_filename"].asString();

        const Json::Value &attr_dtype_map = value["attr_dtype_map"];
        attr_dtype_map_.clear();

        std::unordered_map<std::string, core::Dtype> map_str_dtype = {
                {"UInt16", core::Dtype::UInt16},
                {"Float32", core::Dtype::Float32}};
        auto member_names = attr_dtype_map.getMemberNames();
        for (auto member_name : member_names) {
            std::string key = member_name;
            std::string val = attr_dtype_map[key].asString();
            attr_dtype_map_[key] = map_str_dtype[val];
            utility::LogDebug("{} -> {}", key, val);
        }
        return true;
    }

    float voxel_size_;
    float sdf_trunc_;

    int block_resolution_;
    int block_count_;

    std::string device_;
    std::string hashmap_filename_;

    std::unordered_map<std::string, core::Dtype> attr_dtype_map_;
};

std::shared_ptr<geometry::TSDFVoxelGrid> CreateTSDFVoxelGridFromFile(
        const std::string &filename) {
    auto voxel_grid_ptr = std::make_shared<geometry::TSDFVoxelGrid>();
    ReadTSDFVoxelGrid(filename, *voxel_grid_ptr);
    return voxel_grid_ptr;
}

bool ReadTSDFVoxelGrid(const std::string &filename,
                       geometry::TSDFVoxelGrid &tsdf_voxelgrid) {
    TSDFVoxelGridMetadata metadata;
    bool success = open3d::io::ReadIJsonConvertible(filename, metadata);

    if (!success) {
        utility::LogError("Unable to read TSDFVoxelGrid's metadata!");
    }

    std::string parent_dir =
            utility::filesystem::GetFileParentDirectory(filename);
    if (parent_dir == "") {
        parent_dir = ".";
    }

    std::string filename_prefix = parent_dir + "/" + metadata.hashmap_filename_;
    auto device = core::Device(metadata.device_);
    tsdf_voxelgrid = geometry::TSDFVoxelGrid(
            metadata.attr_dtype_map_, metadata.voxel_size_, metadata.sdf_trunc_,
            metadata.block_resolution_, metadata.block_count_, device);

    auto keys = core::Tensor::Load(filename_prefix + ".key.npy").To(device);
    auto values = core::Tensor::Load(filename_prefix + ".value.npy").To(device);

    core::Tensor addrs, masks;
    tsdf_voxelgrid.GetBlockHashmap()->Insert(keys, values, addrs, masks);

    return true;
}

bool WriteTSDFVoxelGrid(const std::string &filename,
                        const geometry::TSDFVoxelGrid &tsdf_voxelgrid) {
    TSDFVoxelGridMetadata metadata(tsdf_voxelgrid);
    auto extension = utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto filename_noext =
            utility::filesystem::GetFileNameWithoutExtension(filename);

    metadata.hashmap_filename_ = filename_noext + ".hash";
    bool success = open3d::io::WriteIJsonConvertibleToJSON(filename, metadata);
    if (!success) {
        utility::LogError("Unable to write TSDFVoxelGrid's metadata!");
    }

    WriteHashmap(metadata.hashmap_filename_, *tsdf_voxelgrid.GetBlockHashmap());

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d

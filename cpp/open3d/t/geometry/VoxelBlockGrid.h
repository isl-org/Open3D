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

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {

/// A voxel block grid is a sparse grid of voxel blocks.
/// Each voxel block is a dense 3D array, preserving local data distribution.
/// If the block_resolution is set to 1, then the VoxelBlockGrid degenerates to
/// a sparse voxel grid.
class VoxelBlockGrid {
public:
    /// \brief Default Constructor.
    /// Example:
    /// VoxelBlockGrid({"tsdf", "weight", "color"},
    ///                {core::Float32, core::Float32, core::UInt16},
    ///                {{1}, {1}, {3}},
    ///                0.005,
    ///                16,
    ///                core::Device("CUDA:0"),
    ///                core::HashBackendType::Default);
    VoxelBlockGrid(const std::vector<std::string> &attr_names,
                   const std::vector<core::Dtype> &attr_dtypes,
                   const std::vector<core::SizeVector> &attr_channels,
                   double voxel_size,
                   int64_t block_resolution = 16,
                   int64_t block_count = 10000,
                   const core::Device &device = core::Device("CPU:0"),
                   const core::HashBackendType &backend =
                           core::HashBackendType::Default);

    /// Default destructor.
    ~VoxelBlockGrid() = default;

    /// Get the underlying hash map that stores values in structure of arrays
    /// (SoA).
    core::HashMap GetHashMap() { return *block_hashmap_; }

    /// Get the attribute tensor corresponding to the attribute name.
    /// A sugar for hashmap.GetValueTensor(i)
    core::Tensor GetAttribute(const std::string &attr_name) const;

    /// Get the coordinate tensor of active voxels per block, in the shape of
    /// (hashmap.Size(), resolution, resolution, resolution, 3) in Int32.
    /// Useful for developing user-defined tensor-based operations.
    ///
    /// To access actual metric coordinates, multiply by (voxel_size_ *
    /// resolution).
    ///
    /// Example:
    /// For a voxel block grid with (2, 2, 2) block resolution,
    /// if the active block coordinates are {(-1, 3, 2), (0, 2, 4)},
    /// the returned results will be 2 x 8 3D coordinates:
    /// {
    /// (-1, 3, 2) * 2 + (0, 0, 0) = (-2, 6, 4)
    /// (-1, 3, 2) * 2 + (1, 0, 0) = (-1, 6, 4)
    /// ...
    /// (-1, 3, 2) * 2 + (1, 1, 1) = (-1, 7, 5)
    ///
    /// (0, 2, 4) * 2 + (0, 0, 0) = (0, 4, 8)
    /// (0, 2, 4) * 2 + (1, 0, 0) = (1, 4, 8)
    /// ...
    /// (0, 2, 4) * 2 + (1, 1, 1) = (1, 5, 9)
    /// }
    core::Tensor GetVoxelCoordinates(const core::Tensor &voxel_indices) const;

    /// Get a (4, N) index tensor of active voxels.
    /// To access actual metric coordinates, multiply by (voxel_size_ *
    /// resolution).
    core::Tensor GetVoxelIndices() const;

    ////////////////////////////////////////////////////////
    /// Integration related properties
    /// Fuse an RGB-D frame with TSDF integration.
    void Integrate(const Image &depth,
                   const Image &color,
                   const core::Tensor &intrinsics,
                   const core::Tensor &extrinsics,
                   float depth_scale = 1000.0f,
                   float depth_max = 3.0f);

    PointCloud ExtractSurfacePoints(int estimate_number = -1,
                                    float weight_threshold = 3.0f);

    std::unordered_map<std::string, core::Tensor> RayCast(
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics,
            int width,
            int height,
            float depth_scale = 1000.0f,
            float depth_min = 0.1f,
            float depth_max = 3.0f,
            float weight_threshold = 3.0f);

private:
    float voxel_size_;
    int64_t block_resolution_;

    core::Tensor active_block_coords_;

    // Global hash map: 3D coords -> voxel blocks in SoA.
    std::shared_ptr<core::HashMap> block_hashmap_;

    // Local hash map: 3D coords -> indices in block_hashmap_.
    std::shared_ptr<core::HashMap> point_hashmap_;

    // Map: attribute name -> index to access the attribute in SoA.
    std::unordered_map<std::string, int> name_attr_map_;
};
}  // namespace geometry
}  // namespace t
}  // namespace open3d

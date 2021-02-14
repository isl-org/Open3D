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

#include "open3d/t/pipelines/slac/ControlGrid.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

const std::string ControlGrid::kAttrNbGridIdx = "nb_grid_indices";
const std::string ControlGrid::kAttrNbGridPointInterp =
        "nb_grid_point_interp_ratios";
const std::string ControlGrid::kAttrNbGridNormalInterp =
        "nb_grid_normal_interp_ratios";

ControlGrid::ControlGrid(float grid_size,
                         int64_t grid_count,
                         const core::Device& device)
    : grid_size_(grid_size), device_(device) {
    // Maps a coordinate to its non-rigidly transformed coordinate
    ctr_hashmap_ = std::make_shared<core::Hashmap>(
            grid_count, core::Dtype::Int32, core::Dtype::Float32,
            core::SizeVector{3}, core::SizeVector{3}, device);
}

void ControlGrid::Touch(const geometry::PointCloud& pcd) {
    core::Tensor pts = pcd.GetPoints();
    int64_t n = pts.GetLength();

    // Coordinate in the grid unit.
    core::Tensor vals = (pts / grid_size_).Floor();
    core::Tensor keys = vals.To(core::Dtype::Int32);

    // Prepare for insertion with 8 neighbors
    core::Tensor keys_nb({8, n, 3}, core::Dtype::Int32, device_);
    core::Tensor vals_nb({8, n, 3}, core::Dtype::Float32, device_);
    for (int nb = 0; nb < 8; ++nb) {
        int x_sel = (nb & 4) >> 2;
        int y_sel = (nb & 2) >> 1;
        int z_sel = (nb & 1);

        core::Tensor dt = core::Tensor(std::vector<int>{x_sel, y_sel, z_sel},
                                       {1, 3}, core::Dtype::Int32, device_);
        keys_nb[nb] = keys + dt;
        vals_nb[nb] = vals + dt.To(core::Dtype::Float32);
    }
    keys_nb = keys_nb.View({8 * n, 3});

    // Convert back to the meter unit.
    vals_nb = vals_nb.View({8 * n, 3}) * grid_size_;

    core::Tensor addrs_nb, masks_nb;
    ctr_hashmap_->Insert(keys_nb, vals_nb, addrs_nb, masks_nb);
    utility::LogInfo("n * 8 = {}, Hashmap size: {}, capacity: {}, bucket: {}",
                     n * 8, ctr_hashmap_->Size(), ctr_hashmap_->GetCapacity(),
                     ctr_hashmap_->GetBucketCount());
}

void ControlGrid::Compactify() {
    ctr_hashmap_->Rehash(ctr_hashmap_->Size());
    core::Tensor active_addrs;
    ctr_hashmap_->GetActiveIndices(active_addrs);

    utility::LogInfo("active_addrs = {}", active_addrs.ToString());
}

std::tuple<core::Tensor, core::Tensor, core::Tensor>
ControlGrid::GetNeighborGridMap() {
    core::Tensor active_addrs;
    ctr_hashmap_->GetActiveIndices(active_addrs);
    core::Tensor active_indices = active_addrs.To(core::Dtype::Int64);
    core::Tensor active_keys =
            ctr_hashmap_->GetKeyTensor().IndexGet({active_indices});

    int64_t n = active_indices.GetLength();
    core::Tensor keys_nb({6, n, 3}, core::Dtype::Int32, device_);

    core::Tensor dx = core::Tensor(std::vector<int>{1, 0, 0}, {1, 3},
                                   core::Dtype::Int32, device_);
    core::Tensor dy = core::Tensor(std::vector<int>{0, 1, 0}, {1, 3},
                                   core::Dtype::Int32, device_);
    core::Tensor dz = core::Tensor(std::vector<int>{0, 0, 1}, {1, 3},
                                   core::Dtype::Int32, device_);
    keys_nb[0] = active_keys - dx;
    keys_nb[1] = active_keys + dx;
    keys_nb[2] = active_keys - dy;
    keys_nb[3] = active_keys + dy;
    keys_nb[4] = active_keys - dz;
    keys_nb[5] = active_keys + dz;

    // Obtain nearest neighbors
    keys_nb = keys_nb.View({6 * n, 3});

    core::Tensor addrs_nb, masks_nb;
    ctr_hashmap_->Find(keys_nb, addrs_nb, masks_nb);
    return std::make_tuple(active_addrs, addrs_nb.View({6, n}).T().Contiguous(),
                           masks_nb.View({6, n}).T().Contiguous());
}

geometry::PointCloud ControlGrid::Parameterize(
        const geometry::PointCloud& pcd) {
    core::Tensor pts = pcd.GetPoints();
    core::Tensor nms = pcd.GetPointNormals().T().Contiguous();
    int64_t n = pts.GetLength();

    core::Tensor pts_quantized = pts / grid_size_;
    core::Tensor pts_quantized_floor = pts_quantized.Floor();

    // (N x 3) -> [0, 1] for trilinear interpolation
    core::Tensor residual = pts_quantized - pts_quantized_floor;
    std::vector<std::vector<core::Tensor>> residuals(3);
    for (int axis = 0; axis < 3; ++axis) {
        core::Tensor residual_axis = residual.GetItem(
                {core::TensorKey::Slice(core::None, core::None, core::None),
                 core::TensorKey::Index(axis)});

        residuals[axis].emplace_back(1.f - residual_axis);
        residuals[axis].emplace_back(residual_axis);
    }

    core::Tensor keys = pts_quantized_floor.To(core::Dtype::Int32);
    core::Tensor keys_nb({8, n, 3}, core::Dtype::Int32, device_);
    core::Tensor point_ratios_nb({8, n}, core::Dtype::Float32, device_);
    core::Tensor normal_ratios_nb({8, n}, core::Dtype::Float32, device_);
    for (int nb = 0; nb < 8; ++nb) {
        int x_sel = (nb & 4) >> 2;
        int y_sel = (nb & 2) >> 1;
        int z_sel = (nb & 1);

        float x_sign = x_sel * 2.0 - 1.0;
        float y_sign = y_sel * 2.0 - 1.0;
        float z_sign = z_sel * 2.0 - 1.0;

        core::Tensor dt = core::Tensor(std::vector<int>{x_sel, y_sel, z_sel},
                                       {1, 3}, core::Dtype::Int32, device_);
        keys_nb[nb] = keys + dt;
        point_ratios_nb[nb] =
                residuals[0][x_sel] * residuals[1][y_sel] * residuals[2][z_sel];
        normal_ratios_nb[nb] =
                x_sign * nms[0] * residuals[1][y_sel] * residuals[2][z_sel] +
                y_sign * nms[1] * residuals[0][x_sel] * residuals[2][z_sel] +
                z_sign * nms[2] * residuals[0][x_sel] * residuals[1][y_sel];
    }

    keys_nb = keys_nb.View({8 * n, 3});

    core::Tensor addrs_nb, masks_nb;
    ctr_hashmap_->Find(keys_nb, addrs_nb, masks_nb);

    int64_t valid_sum =
            masks_nb.To(core::Dtype::Int64).Sum({0}).Item<int64_t>();
    if (valid_sum != 8 * n) {
        utility::LogError("Unexpected invalid masks exist {} vs {}!", valid_sum,
                          8 * n);
    }

    geometry::PointCloud pcd_with_params = pcd;
    pcd_with_params.SetPointAttr(kAttrNbGridIdx,
                                 addrs_nb.View({8, n}).T().Contiguous());
    pcd_with_params.SetPointAttr(kAttrNbGridPointInterp,
                                 point_ratios_nb.T().Contiguous());
    pcd_with_params.SetPointAttr(kAttrNbGridNormalInterp,
                                 normal_ratios_nb.T().Contiguous());

    return pcd_with_params;
}

geometry::PointCloud ControlGrid::Warp(const geometry::PointCloud& pcd) {
    if (!pcd.HasPointAttr(kAttrNbGridIdx) ||
        !pcd.HasPointAttr(kAttrNbGridPointInterp) ||
        !pcd.HasPointAttr(kAttrNbGridNormalInterp)) {
        utility::LogError(
                "Please use ControlGrid.Parameterize to obtain neighbor grids "
                "before calling Warp");
    }

    // N x 3
    core::Tensor grid_positions = ctr_hashmap_->GetValueTensor();

    // N x 8, we have ensured that every neighbor is valid through grid.Touch()
    core::Tensor nb_grid_indices =
            pcd.GetPointAttr(kAttrNbGridIdx).To(core::Dtype::Int64);
    core::Tensor nb_grid_positions =
            grid_positions.IndexGet({nb_grid_indices.View({-1})})
                    .View({-1, 8, 3});

    // N x 8 x 3 => N x 3 position interpolation
    core::Tensor nb_grid_point_interp =
            pcd.GetPointAttr(kAttrNbGridPointInterp);
    core::Tensor interp_positions =
            (nb_grid_positions * nb_grid_point_interp.View({-1, 8, 1}))
                    .Sum({1});

    // N x 8 x 3 => N x 3 normal interpolation
    core::Tensor nb_grid_normal_interp =
            pcd.GetPointAttr(kAttrNbGridNormalInterp);
    core::Tensor interp_normals =
            (nb_grid_positions * nb_grid_normal_interp.View({-1, 8, 1}))
                    .Sum({1});
    core::Tensor interp_normals_len =
            (interp_normals * interp_normals).Sum({1}).Sqrt();
    interp_normals = interp_normals / interp_normals_len.View({-1, 1});

    geometry::PointCloud interp_pcd(interp_positions);
    interp_pcd.SetPointNormals(interp_normals);
    return interp_pcd;
}

geometry::Image ControlGrid::Warp(const geometry::Image& depth,
                                  const core::Tensor& intrinsics,
                                  const core::Tensor& extrinsics) {
    geometry::PointCloud pcd = geometry::PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics);
    geometry::PointCloud pcd_param = Parameterize(pcd);
    geometry::PointCloud pcd_warped = Warp(pcd_param);
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

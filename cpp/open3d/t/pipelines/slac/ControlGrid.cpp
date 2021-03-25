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

#include "open3d/core/EigenConverter.h"

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
    ctr_hashmap_ = std::make_shared<core::Hashmap>(
            grid_count, core::Dtype::Int32, core::Dtype::Float32,
            core::SizeVector{3}, core::SizeVector{3}, device);
}

ControlGrid::ControlGrid(float grid_size,
                         const core::Tensor& keys,
                         const core::Tensor& values,
                         const core::Device& device)
    : grid_size_(grid_size), device_(device) {
    ctr_hashmap_ = std::make_shared<core::Hashmap>(
            keys.GetLength(), core::Dtype::Int32, core::Dtype::Float32,
            core::SizeVector{3}, core::SizeVector{3}, device);

    core::Tensor addrs, masks;
    ctr_hashmap_->Insert(keys, values, addrs, masks);
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

    // Select anchor point
    core::Tensor active_keys = ctr_hashmap_->GetKeyTensor().IndexGet(
            {active_addrs.To(core::Dtype::Int64)});

    std::vector<Eigen::Vector3i> active_keys_vec =
            core::eigen_converter::TensorToEigenVector3iVector(active_keys);

    std::vector<Eigen::Vector4i> active_keys_indexed(active_keys_vec.size());
    for (size_t i = 0; i < active_keys_vec.size(); ++i) {
        active_keys_indexed[i](0) = active_keys_vec[i](0);
        active_keys_indexed[i](1) = active_keys_vec[i](1);
        active_keys_indexed[i](2) = active_keys_vec[i](2);
        active_keys_indexed[i](3) = i;
    }

    std::sort(active_keys_indexed.begin(), active_keys_indexed.end(),
              [=](const Eigen::Vector4i& a, const Eigen::Vector4i& b) -> bool {
                  return (a(2) < b(2)) || (a(2) == b(2) && a(1) < b(1)) ||
                         (a(2) == b(2) && a(1) == b(1) && a(0) < b(0));
              });
    anchor_idx_ =
            active_addrs[active_keys_indexed[active_keys_indexed.size() / 2](3)]
                    .Item<int>();
    utility::LogInfo("{}",
                     ctr_hashmap_->GetKeyTensor()[anchor_idx_].ToString());
    std::cout << anchor_idx_ << "\n";
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
    core::Tensor nms;
    if (pcd.HasPointNormals()) {
        nms = pcd.GetPointNormals().T().Contiguous();
    }
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
        if (pcd.HasPointNormals()) {
            normal_ratios_nb[nb] =
                    x_sign * nms[0] * residuals[1][y_sel] *
                            residuals[2][z_sel] +
                    y_sign * nms[1] * residuals[0][x_sel] *
                            residuals[2][z_sel] +
                    z_sign * nms[2] * residuals[0][x_sel] * residuals[1][y_sel];
        }
    }

    keys_nb = keys_nb.View({8 * n, 3});

    core::Tensor addrs_nb, masks_nb;
    ctr_hashmap_->Find(keys_nb, addrs_nb, masks_nb);

    // (n, 8)
    addrs_nb = addrs_nb.View({8, n}).T().Contiguous();
    // (n, 8)
    point_ratios_nb = point_ratios_nb.T().Contiguous();

    /// TODO: allow entries with less than 8 neighbors, probably in a
    /// kernel. Now we simply discard them and only accepts points with all
    /// 8 neighbors in the control grid map.
    core::Tensor valid_mask =
            masks_nb.View({8, n}).To(core::Dtype::Int64).Sum({0}).Eq(8);

    geometry::PointCloud pcd_with_params = pcd;
    pcd_with_params.SetPoints(pcd.GetPoints().IndexGet({valid_mask}));
    pcd_with_params.SetPointAttr(kAttrNbGridIdx,
                                 addrs_nb.IndexGet({valid_mask}));
    pcd_with_params.SetPointAttr(kAttrNbGridPointInterp,
                                 point_ratios_nb.IndexGet({valid_mask}));

    if (pcd.HasPointColors()) {
        pcd_with_params.SetPointColors(
                pcd.GetPointColors().IndexGet({valid_mask}));
    }
    if (pcd.HasPointNormals()) {
        pcd_with_params.SetPointNormals(
                pcd.GetPointNormals().IndexGet({valid_mask}));
        normal_ratios_nb = normal_ratios_nb.T().Contiguous();
        pcd_with_params.SetPointAttr(kAttrNbGridNormalInterp,
                                     normal_ratios_nb.IndexGet({valid_mask}));
    }

    return pcd_with_params;
}

geometry::PointCloud ControlGrid::Warp(const geometry::PointCloud& pcd) {
    if (!pcd.HasPointAttr(kAttrNbGridIdx) ||
        !pcd.HasPointAttr(kAttrNbGridPointInterp)) {
        utility::LogError(
                "Please use ControlGrid.Parameterize to obtain attributes "
                "regarding neighbor grids before calling Warp");
    }

    // N x 3
    core::Tensor grid_positions = ctr_hashmap_->GetValueTensor();

    // N x 8, we have ensured that every neighbor is valid through
    // grid.Parameterize
    core::Tensor nb_grid_indices =
            pcd.GetPointAttr(kAttrNbGridIdx).To(core::Dtype::Int64);
    core::Tensor nb_grid_positions =
            grid_positions.IndexGet({nb_grid_indices.View({-1})})
                    .View({-1, 8, 3});

    // (N, 8, 3) x (N, 8, 1) => Reduce on dim 1 => (N, 3) position
    // interpolation.
    core::Tensor nb_grid_point_interp =
            pcd.GetPointAttr(kAttrNbGridPointInterp);
    core::Tensor interp_positions =
            (nb_grid_positions * nb_grid_point_interp.View({-1, 8, 1}))
                    .Sum({1});

    geometry::PointCloud interp_pcd(interp_positions);

    if (pcd.HasPointNormals()) {
        // (N, 8, 3) x (N, 8, 1) => Reduce on dim 1 => (N, 3) normal
        // interpolation.
        core::Tensor nb_grid_normal_interp =
                pcd.GetPointAttr(kAttrNbGridNormalInterp);
        core::Tensor interp_normals =
                (nb_grid_positions * nb_grid_normal_interp.View({-1, 8, 1}))
                        .Sum({1});
        core::Tensor interp_normals_len =
                (interp_normals * interp_normals).Sum({1}).Sqrt();
        interp_normals = interp_normals / interp_normals_len.View({-1, 1});
        interp_pcd.SetPointNormals(interp_normals);
    }

    if (pcd.HasPointColors()) {
        interp_pcd.SetPointColors(pcd.GetPointColors());
    }
    return interp_pcd;
}

geometry::Image ControlGrid::Warp(const geometry::Image& depth,
                                  const core::Tensor& intrinsics,
                                  const core::Tensor& extrinsics,
                                  float depth_scale,
                                  float depth_max) {
    geometry::PointCloud pcd = geometry::PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics, depth_scale, depth_max);

    geometry::PointCloud pcd_param = Parameterize(pcd);
    geometry::PointCloud pcd_warped = Warp(pcd_param);

    return geometry::Image(
            pcd_warped
                    .ProjectDepth(depth.GetCols(), depth.GetRows(), intrinsics,
                                  extrinsics, depth_scale, depth_max)
                    .AsTensor()
                    .To(core::Dtype::UInt16));
}

std::pair<geometry::Image, geometry::Image> ControlGrid::Warp(
        const geometry::Image& depth,
        const geometry::Image& color,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max) {
    geometry::PointCloud pcd = geometry::PointCloud::CreateFromRGBDImage(
            geometry::RGBDImage(color, depth), intrinsics, extrinsics,
            depth_scale, depth_max);

    geometry::PointCloud pcd_param = Parameterize(pcd);
    geometry::PointCloud pcd_warped = Warp(pcd_param);

    auto rgbd_warped =
            pcd_warped.ProjectRGBD(depth.GetCols(), depth.GetRows(), intrinsics,
                                   extrinsics, depth_scale, depth_max);

    return std::make_pair(geometry::Image(rgbd_warped.first.AsTensor().To(
                                  core::Dtype::UInt16)),
                          geometry::Image(rgbd_warped.second.AsTensor()));
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

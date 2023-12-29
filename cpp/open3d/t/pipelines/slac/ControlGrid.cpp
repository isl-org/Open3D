// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/slac/ControlGrid.h"

#include "open3d/core/EigenConverter.h"
#include "open3d/core/hashmap/HashSet.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

const std::string ControlGrid::kGrid8NbIndices = "Grid8NbIndices";
const std::string ControlGrid::kGrid8NbVertexInterpRatios =
        "Grid8NbVertexInterpRatios";
const std::string ControlGrid::kGrid8NbNormalInterpRatios =
        "Grid8NbNormalInterpRatios";

ControlGrid::ControlGrid(float grid_size,
                         int64_t grid_count,
                         const core::Device& device)
    : grid_size_(grid_size), device_(device) {
    ctr_hashmap_ = std::make_shared<core::HashMap>(
            grid_count, core::Int32, core::SizeVector{3}, core::Float32,
            core::SizeVector{3}, device);
}

ControlGrid::ControlGrid(float grid_size,
                         const core::Tensor& keys,
                         const core::Tensor& values,
                         const core::Device& device)
    : grid_size_(grid_size), device_(device) {
    ctr_hashmap_ = std::make_shared<core::HashMap>(
            2 * keys.GetLength(), core::Int32, core::SizeVector{3},
            core::Float32, core::SizeVector{3}, device);

    core::Tensor buf_indices, masks;
    ctr_hashmap_->Insert(keys, values, buf_indices, masks);
}

void ControlGrid::Touch(const geometry::PointCloud& pcd) {
    core::Tensor pts = pcd.GetPointPositions();
    int64_t n = pts.GetLength();

    // Coordinate in the grid unit.
    core::Tensor vals = (pts / grid_size_).Floor();
    core::Tensor keys = vals.To(core::Int32);

    // Prepare for insertion with 8 neighbors
    core::Tensor keys_nb({8, n, 3}, core::Int32, device_);
    core::Tensor vals_nb({8, n, 3}, core::Float32, device_);
    for (int nb = 0; nb < 8; ++nb) {
        int x_sel = (nb & 4) >> 2;
        int y_sel = (nb & 2) >> 1;
        int z_sel = (nb & 1);

        core::Tensor dt = core::Tensor(std::vector<int>{x_sel, y_sel, z_sel},
                                       {1, 3}, core::Int32, device_);
        keys_nb[nb] = keys + dt;
        vals_nb[nb] = vals + dt.To(core::Float32);
    }
    keys_nb = keys_nb.View({8 * n, 3});

    // Convert back to the meter unit.
    vals_nb = vals_nb.View({8 * n, 3}) * grid_size_;

    core::HashSet unique_hashset(n, core::Int32, core::SizeVector{3}, device_);

    core::Tensor buf_indices_unique, masks_unique;
    unique_hashset.Insert(keys_nb, buf_indices_unique, masks_unique);

    core::Tensor buf_indices, masks;
    ctr_hashmap_->Insert(keys_nb.IndexGet({masks_unique}),
                         vals_nb.IndexGet({masks_unique}), buf_indices, masks);
}

void ControlGrid::Compactify() {
    ctr_hashmap_->Reserve(ctr_hashmap_->Size() * 2);

    core::Tensor active_buf_indices;
    ctr_hashmap_->GetActiveIndices(active_buf_indices);

    // Select anchor point
    core::Tensor active_keys = ctr_hashmap_->GetKeyTensor().IndexGet(
            {active_buf_indices.To(core::Int64)});

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
            active_buf_indices[active_keys_indexed[active_keys_indexed.size() /
                                                   2](3)]
                    .Item<int>();
}

std::tuple<core::Tensor, core::Tensor, core::Tensor>
ControlGrid::GetNeighborGridMap() {
    core::Tensor active_buf_indices;
    ctr_hashmap_->GetActiveIndices(active_buf_indices);

    core::Tensor active_indices = active_buf_indices.To(core::Int64);
    core::Tensor active_keys =
            ctr_hashmap_->GetKeyTensor().IndexGet({active_indices});

    int64_t n = active_indices.GetLength();
    core::Tensor keys_nb({6, n, 3}, core::Int32, device_);

    core::Tensor dx = core::Tensor(std::vector<int>{1, 0, 0}, {1, 3},
                                   core::Int32, device_);
    core::Tensor dy = core::Tensor(std::vector<int>{0, 1, 0}, {1, 3},
                                   core::Int32, device_);
    core::Tensor dz = core::Tensor(std::vector<int>{0, 0, 1}, {1, 3},
                                   core::Int32, device_);
    keys_nb[0] = active_keys - dx;
    keys_nb[1] = active_keys + dx;
    keys_nb[2] = active_keys - dy;
    keys_nb[3] = active_keys + dy;
    keys_nb[4] = active_keys - dz;
    keys_nb[5] = active_keys + dz;

    // Obtain nearest neighbors
    keys_nb = keys_nb.View({6 * n, 3});

    core::Tensor buf_indices_nb, masks_nb;
    ctr_hashmap_->Find(keys_nb, buf_indices_nb, masks_nb);

    return std::make_tuple(active_buf_indices,
                           buf_indices_nb.View({6, n}).T().Contiguous(),
                           masks_nb.View({6, n}).T().Contiguous());
}

geometry::PointCloud ControlGrid::Parameterize(
        const geometry::PointCloud& pcd) {
    core::Tensor pts = pcd.GetPointPositions();
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

    core::Tensor keys = pts_quantized_floor.To(core::Int32);
    core::Tensor keys_nb({8, n, 3}, core::Int32, device_);
    core::Tensor point_ratios_nb({8, n}, core::Float32, device_);
    core::Tensor normal_ratios_nb({8, n}, core::Float32, device_);
    for (int nb = 0; nb < 8; ++nb) {
        int x_sel = (nb & 4) >> 2;
        int y_sel = (nb & 2) >> 1;
        int z_sel = (nb & 1);

        float x_sign = x_sel * 2.0 - 1.0;
        float y_sign = y_sel * 2.0 - 1.0;
        float z_sign = z_sel * 2.0 - 1.0;

        core::Tensor dt = core::Tensor(std::vector<int>{x_sel, y_sel, z_sel},
                                       {1, 3}, core::Int32, device_);
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

    core::Tensor buf_indices_nb, masks_nb;
    ctr_hashmap_->Find(keys_nb, buf_indices_nb, masks_nb);

    // (n, 8)
    buf_indices_nb = buf_indices_nb.View({8, n}).T().Contiguous();
    // (n, 8)
    point_ratios_nb = point_ratios_nb.T().Contiguous();

    /// TODO(wei): allow entries with less than 8 neighbors, probably in a
    /// kernel. Now we simply discard them and only accepts points with all
    /// 8 neighbors in the control grid map.
    core::Tensor valid_mask =
            masks_nb.View({8, n}).To(core::Int64).Sum({0}).Eq(8);

    geometry::PointCloud pcd_with_params = pcd;
    pcd_with_params.SetPointPositions(
            pcd.GetPointPositions().IndexGet({valid_mask}));
    pcd_with_params.SetPointAttr(kGrid8NbIndices,
                                 buf_indices_nb.IndexGet({valid_mask}));
    pcd_with_params.SetPointAttr(kGrid8NbVertexInterpRatios,
                                 point_ratios_nb.IndexGet({valid_mask}));

    if (pcd.HasPointColors()) {
        pcd_with_params.SetPointColors(
                pcd.GetPointColors().IndexGet({valid_mask}));
    }
    if (pcd.HasPointNormals()) {
        pcd_with_params.SetPointNormals(
                pcd.GetPointNormals().IndexGet({valid_mask}));
        normal_ratios_nb = normal_ratios_nb.T().Contiguous();
        pcd_with_params.SetPointAttr(kGrid8NbNormalInterpRatios,
                                     normal_ratios_nb.IndexGet({valid_mask}));
    }

    return pcd_with_params;
}

geometry::PointCloud ControlGrid::Deform(const geometry::PointCloud& pcd) {
    if (!pcd.HasPointAttr(kGrid8NbIndices) ||
        !pcd.HasPointAttr(kGrid8NbVertexInterpRatios)) {
        utility::LogError(
                "Please use ControlGrid.Parameterize to obtain attributes "
                "regarding neighbor grids before calling Deform");
    }

    // N x 3
    core::Tensor grid_positions = ctr_hashmap_->GetValueTensor();

    // N x 8, we have ensured that every neighbor is valid through
    // grid.Parameterize
    core::Tensor nb_grid_indices =
            pcd.GetPointAttr(kGrid8NbIndices).To(core::Int64);
    core::Tensor nb_grid_positions =
            grid_positions.IndexGet({nb_grid_indices.View({-1})})
                    .View({-1, 8, 3});

    // (N, 8, 3) x (N, 8, 1) => Reduce on dim 1 => (N, 3) position
    // interpolation.
    core::Tensor nb_grid_point_interp =
            pcd.GetPointAttr(kGrid8NbVertexInterpRatios);
    core::Tensor interp_positions =
            (nb_grid_positions * nb_grid_point_interp.View({-1, 8, 1}))
                    .Sum({1});

    geometry::PointCloud interp_pcd(interp_positions);

    if (pcd.HasPointNormals()) {
        // (N, 8, 3) x (N, 8, 1) => Reduce on dim 1 => (N, 3) normal
        // interpolation.
        core::Tensor nb_grid_normal_interp =
                pcd.GetPointAttr(kGrid8NbNormalInterpRatios);
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

geometry::Image ControlGrid::Deform(const geometry::Image& depth,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& extrinsics,
                                    float depth_scale,
                                    float depth_max) {
    geometry::PointCloud pcd = geometry::PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics, depth_scale, depth_max);

    geometry::PointCloud pcd_param = Parameterize(pcd);
    geometry::PointCloud pcd_deformed = Deform(pcd_param);

    return pcd_deformed.ProjectToDepthImage(depth.GetCols(), depth.GetRows(),
                                            intrinsics, extrinsics, depth_scale,
                                            depth_max);
}

geometry::RGBDImage ControlGrid::Deform(const geometry::RGBDImage& rgbd,
                                        const core::Tensor& intrinsics,
                                        const core::Tensor& extrinsics,
                                        float depth_scale,
                                        float depth_max) {
    geometry::PointCloud pcd = geometry::PointCloud::CreateFromRGBDImage(
            rgbd, intrinsics, extrinsics, depth_scale, depth_max);

    geometry::PointCloud pcd_param = Parameterize(pcd);
    geometry::PointCloud pcd_deformed = Deform(pcd_param);

    int cols = rgbd.depth_.GetCols();
    int rows = rgbd.color_.GetRows();

    return pcd_deformed.ProjectToRGBDImage(cols, rows, intrinsics, extrinsics,
                                           depth_scale, depth_max);
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

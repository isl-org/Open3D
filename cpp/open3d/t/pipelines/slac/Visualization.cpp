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

#include "open3d/t/pipelines/slac/Visualization.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

inline Eigen::Vector3d Jet(double v, double vmin, double vmax) {
    Eigen::Vector3d c(1, 1, 1);
    double dv;

    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c(0) = 0;
        c(1) = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c(0) = 0;
        c(2) = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c(0) = 4 * (v - vmin - 0.5 * dv) / dv;
        c(2) = 0;
    } else {
        c(1) = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c(2) = 0;
    }

    return c;
}

t::geometry::PointCloud CreateTPCDFromFile(const std::string& fname,
                                           const core::Device& device) {
    auto pcd = io::CreatePointCloudFromFile(fname);
    return t::geometry::PointCloud::FromLegacyPointCloud(
            *pcd, core::Dtype::Float32, device);
}

void VisualizePCDCorres(t::geometry::PointCloud& tpcd_param_i,
                        t::geometry::PointCloud& tpcd_param_j,
                        const Eigen::Matrix4d& pose_ij) {
    auto pcd_i = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_param_i.ToLegacyPointCloud());
    auto pcd_j = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_param_j.ToLegacyPointCloud());
    pcd_i->Transform(pose_ij);

    std::vector<std::pair<int, int>> corres_lines;
    for (int64_t i = 0; i < tpcd_param_i.GetPoints().GetLength(); ++i) {
        std::pair<int, int> pair = {i, i};
        corres_lines.push_back(pair);
    }
    auto lineset =
            open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd_i, *pcd_j, corres_lines);
    lineset->PaintUniformColor({0, 1, 0});
    visualization::DrawGeometries({pcd_i, pcd_j, lineset});
}

void VisualizePCDGridCorres(t::geometry::PointCloud& tpcd_param,
                            ControlGrid& ctr_grid) {
    // Prepare all ctr grid point cloud for lineset
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_param.ToLegacyPointCloud());

    t::geometry::PointCloud tpcd_grid(ctr_grid.GetCurrPositions());
    auto pcd_grid = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_grid.ToLegacyPointCloud());

    // Prepare n x 8 corres for visualization
    core::Tensor corres = tpcd_param.GetPointAttr(ControlGrid::kAttrNbGridIdx)
                                  .To(core::Device("CPU:0"));
    std::vector<std::pair<int, int>> corres_lines;
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            std::pair<int, int> pair = {i, corres[i][k].Item<int>()};
            corres_lines.push_back(pair);
        }
    }
    auto lineset =
            open3d::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd, *pcd_grid, corres_lines);

    core::Tensor corres_interp =
            tpcd_param.GetPointAttr(ControlGrid::kAttrNbGridPointInterp)
                    .To(core::Device("CPU:0"));
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            float ratio = corres_interp[i][k].Item<float>();
            Eigen::Vector3d color = Jet(ratio, 0, 0.5);
            lineset->colors_.push_back(color);
        }
    }

    // Prepare nb point cloud for visualization
    t::geometry::PointCloud tpcd_grid_nb(tpcd_grid.GetPoints().IndexGet(
            {corres.View({-1}).To(core::Dtype::Int64)}));
    auto pcd_grid_nb = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_grid_nb.ToLegacyPointCloud());
    pcd_grid_nb->PaintUniformColor({1, 0, 0});

    visualization::DrawGeometries({lineset, pcd, pcd_grid_nb});
}

void VisualizeWarp(const geometry::PointCloud& tpcd_param,
                   ControlGrid& ctr_grid) {
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_param.ToLegacyPointCloud());
    pcd->PaintUniformColor({1, 0, 0});

    auto tpcd_warped = ctr_grid.Warp(tpcd_param);
    auto pcd_warped = std::make_shared<open3d::geometry::PointCloud>(
            tpcd_warped.ToLegacyPointCloud());
    pcd_warped->PaintUniformColor({0, 1, 0});
    visualization::DrawGeometries({pcd, pcd_warped});
}
}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

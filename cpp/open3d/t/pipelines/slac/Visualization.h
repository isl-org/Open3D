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

#pragma once

#include "open3d/core/EigenConverter.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

t::geometry::PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0"));

void VisualizePCDCorres(t::geometry::PointCloud& tpcd_i,
                        t::geometry::PointCloud& tpcd_j,
                        t::geometry::PointCloud& tpcd_param_i,
                        t::geometry::PointCloud& tpcd_param_j,
                        const core::Tensor& Ti,
                        const core::Tensor& Tj);

void VisualizePCDGridCorres(t::geometry::PointCloud& tpcd_param,
                            ControlGrid& ctr_grid,
                            bool show_lines = true);

void VisualizeWarp(const geometry::PointCloud& tpcd_param,
                   ControlGrid& ctr_grid);

void VisualizeRegularizor(ControlGrid& cgrid);
}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

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

#include "open3d/t/pipelines/slac/ControlGrid.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/slac/Visualization.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

static t::geometry::PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0")) {
    auto pcd = io::CreatePointCloudFromFile(fname);
    return t::geometry::PointCloud::FromLegacy(*pcd, core::Float32, device);
}

class ControlGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(ControlGrid,
                         ControlGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

// TODO(wei): more well-designed test cases
TEST_P(ControlGridPermuteDevices, Touch) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);

    t::geometry::PointCloud pcd_param = cgrid.Parameterize(pcd);
}

TEST_P(ControlGridPermuteDevices, Deform) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);
    cgrid.Compactify();

    t::geometry::PointCloud pcd_param = cgrid.Parameterize(pcd);

    core::Tensor prev = cgrid.GetInitPositions();
    core::Tensor curr = cgrid.GetCurrPositions();
    curr[0][0] += 0.5;
    curr[1][2] -= 0.5;
    curr[2][1] += 0.5;
}

TEST_P(ControlGridPermuteDevices, Regularizer) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);
    cgrid.Compactify();
    core::Tensor prev = cgrid.GetInitPositions();
    core::Tensor curr = cgrid.GetCurrPositions();
    curr[0][0] += 0.2;
    curr[1][2] -= 0.2;
    curr[2][1] += 0.2;
}

}  // namespace tests
}  // namespace open3d

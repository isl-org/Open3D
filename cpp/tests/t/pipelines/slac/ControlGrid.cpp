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

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/io/PointCloudIO.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

t::geometry::PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0")) {
    auto pcd = io::CreatePointCloudFromFile(fname);
    return t::geometry::PointCloud::FromLegacyPointCloud(
            *pcd, core::Dtype::Float32, device);
}

class ControlGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(ControlGrid,
                         ControlGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class ControlGridPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        ControlGrid,
        ControlGridPermuteDevicePairs,
        testing::ValuesIn(ControlGridPermuteDevicePairs::TestCases()));

TEST_P(ControlGridPermuteDevices, Touch) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.2, 1000, device);

    t::geometry::PointCloud pcd = CreateTPCDFromFile(
            std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd", device);
    cgrid.Touch(pcd);
}

}  // namespace tests
}  // namespace open3d

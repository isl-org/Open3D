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

#include "open3d/io/tPointCloudIO.h"

#include <gtest/gtest.h>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/tgeometry/PointCloud.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

using open3d::core::Tensor;
using open3d::core::TensorList;
using open3d::io::ReadPointCloud;
using open3d::io::ReadPointCloudOption;
using open3d::io::WritePointCloud;
using open3d::io::WritePointCloudOption;

namespace {

double MaxDistance(const TensorList &a, const TensorList &b) {
    core::SizeVector allDims(a.GetElementShape().size() + 1);
    std::iota(allDims.begin(), allDims.end(), 0);
    // Note: cannot use ASSERT_EQ because we return non-void
    EXPECT_EQ(a.AsTensor().GetShape(), b.AsTensor().GetShape());
    return (a.AsTensor() - b.AsTensor()).Abs().Max(allDims).Item<double>();
}

struct TensorCtorData {
    std::vector<double> values;
    core::SizeVector size;
};

enum class IsAscii : bool { BINARY = false, ASCII = true };
enum class Compressed : bool { UNCOMPRESSED = false, COMPRESSED = true };
struct ReadWritePCArgs {
    std::string filename;
    IsAscii write_ascii;
    Compressed compressed;
    std::unordered_map<std::string, double> attributes_max_errors;
};

}  // namespace

const std::unordered_map<std::string, TensorCtorData> pc_data_1{
        {"points",
         {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
           1.0},
          {5, 3}}},
        {"intensities", {{0, 0.5, 0.5, 0.5, 1.0}, {5, 1}}}},
        // bad data
        pc_data_bad{
                {"points", {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {2, 3}}},
                {"intensities", {{0}, {1, 1}}},
        };

const std::vector<ReadWritePCArgs> pcArgs({
        {"test.xyzi",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"points", 1e-4}, {"intensities", 1e-4}}},  // 0
});

class ReadWriteTPC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWriteTPC, testing::ValuesIn(pcArgs));

TEST_P(ReadWriteTPC, Basic) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU", 0);
    core::Dtype dtype = core::Dtype::Float64;
    tgeometry::PointCloud pc1(dtype, device);

    for (const auto &attr_tensor : pc_data_1) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(attr,
                         TensorList::FromTensor(
                                 {tensor.values, tensor.size, dtype, device}));
    }

    // we loose some precision when saving generated data
    // test writing if we have point, normal, and colors in pc
    EXPECT_TRUE(WritePointCloud(
            args.filename, pc1,
            {bool(args.write_ascii), bool(args.compressed), true}));
    tgeometry::PointCloud pc2(dtype, device);
    EXPECT_TRUE(
            ReadPointCloud(args.filename, pc2, {"auto", false, false, true}));

    for (const auto &attribute_max_err : args.attributes_max_errors) {
        const std::string &attribute = attribute_max_err.first;
        const double max_error = attribute_max_err.second;
        SCOPED_TRACE(attribute);
        EXPECT_LT(MaxDistance(pc1.GetPointAttr(attribute),
                              pc2.GetPointAttr(attribute)),
                  max_error);
    }

    // Loaded data when saved should be identical when reloaded
    EXPECT_TRUE(WritePointCloud(
            args.filename, pc2,
            {bool(args.write_ascii), bool(args.compressed), true}));
    tgeometry::PointCloud pc3(dtype, device);
    EXPECT_TRUE(
            ReadPointCloud(args.filename, pc3, {"auto", false, false, true}));
    for (const auto &attribute_max_err : args.attributes_max_errors) {
        const std::string &attribute = attribute_max_err.first;
        const double max_error = attribute_max_err.second;
        SCOPED_TRACE(attribute);
        EXPECT_EQ(MaxDistance(pc3.GetPointAttr(attribute),
                              pc2.GetPointAttr(attribute)),
                  0);
    }
}

TEST_P(ReadWriteTPC, WriteBadData) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU", 0);
    core::Dtype dtype = core::Dtype::Float64;
    tgeometry::PointCloud pc1(dtype, device);

    for (const auto &attr_tensor : pc_data_bad) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(attr,
                         TensorList::FromTensor(
                                 {tensor.values, tensor.size, dtype, device}));
    }

    EXPECT_FALSE(WritePointCloud(
            args.filename, pc1,
            {bool(args.write_ascii), bool(args.compressed), true}));
}

}  // namespace tests

}  // namespace open3d

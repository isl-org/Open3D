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

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TransformationEstimationPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TransformationEstimation,
                         TransformationEstimationPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TransformationEstimationPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TransformationEstimation,
        TransformationEstimationPermuteDevicePairs,
        testing::ValuesIn(
                TransformationEstimationPermuteDevicePairs::TestCases()));

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPoint) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);

    std::vector<bool> corres_select_bool_vec{true,  true, true, true, true,
                                             true,  true, true, true, true,
                                             false, true, true, true};
    core::Tensor corres_select_bool(corres_select_bool_vec, {14},
                                    core::Dtype::Bool, device);
    std::vector<int64_t> corres_indices_vec{10, 10, 10, 8, 10, 2, 7,
                                            7,  3,  6,  7, 0,  8};
    core::Tensor corres_indices(corres_indices_vec, {13}, core::Dtype::Int64,
                                device);
    t::pipelines::registration::CorrespondenceSet corres =
            std::make_pair(corres_select_bool, corres_indices);

    t::pipelines::registration::TransformationEstimationPointToPoint
            estimation_p2p;
    double p2p_rmse =
            estimation_p2p.ComputeRMSE(source_device, target_device, corres);

    EXPECT_NEAR(p2p_rmse, 0.746223, 0.0001);
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPoint) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);

    std::vector<bool> corres_select_bool_vec{true,  true, true, true, true,
                                             true,  true, true, true, true,
                                             false, true, true, true};
    core::Tensor corres_select_bool(corres_select_bool_vec, {14},
                                    core::Dtype::Bool, device);

    std::vector<int64_t> corres_indices_vec{10, 10, 10, 8, 10, 2, 7,
                                            7,  3,  6,  7, 0,  8};
    core::Tensor corres_indices(corres_indices_vec, {13}, core::Dtype::Int64,
                                device);

    t::pipelines::registration::CorrespondenceSet corres =
            std::make_pair(corres_select_bool, corres_indices);

    t::pipelines::registration::TransformationEstimationPointToPoint
            estimation_p2p;

    // Get transfrom.
    core::Tensor p2p_transform = estimation_p2p.ComputeTransformation(
            source_device, target_device, corres);
    // Apply transform.
    t::geometry::PointCloud source_transformed_p2p = source_device.Clone();
    source_transformed_p2p.Transform(p2p_transform.To(dtype));
    double p2p_rmse_ = estimation_p2p.ComputeRMSE(source_transformed_p2p,
                                                  target_device, corres);

    // Compare the new RMSE after transformation.
    EXPECT_NEAR(p2p_rmse_, 0.545857, 0.0001);
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPlane) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);

    std::vector<float> target_normals_vec{
            -0.0085016, -0.22355,  -0.519574, 0.257463,   -0.0738755, -0.698319,
            0.0574301,  -0.484248, -0.409929, -0.0123503, -0.230172,  -0.52072,
            0.355904,   -0.142007, -0.720467, 0.0674038,  -0.418757,  -0.458602,
            0.226091,   0.258253,  -0.874024, 0.43979,    0.122441,   -0.574998,
            0.109144,   0.180992,  -0.762368, 0.273325,   0.292013,   -0.903111,
            0.385407,   -0.212348, -0.277818};
    core::Tensor target_normals(target_normals_vec, {11, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    std::vector<bool> corres_select_bool_vec{true,  true, true, true, true,
                                             true,  true, true, true, true,
                                             false, true, true, true};
    core::Tensor corres_select_bool(corres_select_bool_vec, {14},
                                    core::Dtype::Bool, device);

    std::vector<int64_t> corres_indices_vec{10, 10, 10, 8, 10, 2, 7,
                                            7,  3,  6,  7, 0,  8};
    core::Tensor corres_indices(corres_indices_vec, {13}, core::Dtype::Int64,
                                device);
    t::pipelines::registration::CorrespondenceSet corres =
            std::make_pair(corres_select_bool, corres_indices);
    t::pipelines::registration::TransformationEstimationPointToPlane
            estimation_p2plane;
    double p2plane_rmse = estimation_p2plane.ComputeRMSE(source_device,
                                                         target_device, corres);

    EXPECT_NEAR(p2plane_rmse, 0.319101, 0.0001);
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPlane) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            0.887091, 2.39347, 1.59234,  1.94008,  2.20845, 1.82707,
            2.66347,  2.22691, 1.68576,  1.46114,  2.40838, 0.799806,
            1.01124,  2.18296, 2.31533,  1.9944,   2.25733, 1.26651,
            2.68741,  2.19797, 1.24086,  3.16343,  2.40136, 0.817764,
            2.66704,  2.39729, 0.773001, 3.79092,  1.89428, 1.25752,
            0.775299, 2.18966, 2.04034,  1.37365,  2.29784, 1.24938,
            1.95926,  2.39706, 0.787228, 0.757318, 1.69401, 2.06456,
            1.45666,  1.57423, 1.19468,  3.78811,  1.55738, 1.30157,
            1.06896,  1.31853, 2.33471,  3.25427,  2.056,   1.13321,
            1.96912,  1.11446, 1.87513,  2.96744,  1.17175, 1.59398,
            1.24847,  2.25907, 2.08582,  1.18295,  1.0485,  2.2624,
            1.22541,  1.50957, 2.25061,  2.60872,  1.69637, 1.54101,
            1.69047,  1.17578, 1.60525,  1.91826,  1.52617, 1.84756,
            0.659388, 1.03491, 2.43713,  2.66351,  1.47928, 1.69352,
            1.09513,  1.14816, 2.35062,  3.73759,  1.17301, 1.29953,
            3.36734,  1.52129, 1.45003,  3.37945,  1.17268, 1.44556,
            1.90626,  1.64181, 1.47832,  2.4974,   1.14841, 1.74182,
            0.690966, 1.30285, 2.38853};
    core::Tensor source_points(src_points_vec, {35, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.86474, 2.54549, 1.24267, 2.40396, 2.42889, 1.17122, 1.76377,
            2.39071, 1.53602, 1.88459, 2.37812, 1.13407, 2.46393, 2.35112,
            1.55615, 2.34309, 2.23165, 1.98089, 1.21182, 1.75949, 1.13272,
            1.15189, 1.32677, 1.18371, 2.49224, 1.81428, 1.73944, 3.11456,
            1.23751, 1.87392, 1.25517, 1.29295, 1.3956,  1.80309, 1.26725,
            1.56453, 3.04598, 1.75961, 1.77977, 1.23481, 1.77748, 1.49404,
            1.82718, 1.788,   1.55078, 3.14181, 1.66811, 2.018,   2.50141,
            1.25167, 1.73528, 1.76483, 2.02266, 1.1409,  2.98848, 2.34816,
            1.58976, 1.25301, 2.34352, 1.54821, 2.80172, 1.80443, 1.98825,
            1.26019, 2.40354, 1.14469};
    core::Tensor target_points(target_points_vec, {22, 3}, dtype, device);

    std::vector<float> target_normals_vec{
            -0.00806126, -0.958624,  -0.228131,  0.0834076,  -0.708154,
            -0.317853,   0.0465067,  -0.703233,  -0.397652,  -0.0816758,
            -0.515562,   -0.311357,  -0.10263,   -0.321624,  -0.428979,
            0.241332,    0.0811388,  -0.800067,  0.774198,   -0.0106762,
            -0.0543447,  0.73333,    -0.0341924, -0.0884099, 0.0271548,
            -0.0867852,  -0.618751,  0.204526,   0.232225,   -0.904236,
            0.354513,    0.198156,   -0.715834,  0.27283,    0.255815,
            -0.919196,   -0.0438041, -0.225404,  -0.470352,  0.438351,
            0.180976,    -0.617539,  0.31526,    -0.0119215, -0.703239,
            0.180907,    0.168824,   -0.790007,  0.189019,   0.247439,
            -0.880691,   -0.0146072, -0.254063,  -0.164816,  0.140616,
            -0.252293,   -0.596847,  0.361736,   -0.249353,  -0.419919,
            0.22753,     0.244338,   -0.927009,  0.536729,   -0.155309,
            -0.246049};

    core::Tensor target_normals(target_normals_vec, {22, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    std::vector<bool> corres_select_bool_vec{
            false, true,  true, true,  false, true, true, true,  true,
            true,  false, true, true,  false, true, true, false, true,
            true,  true,  true, false, true,  true, true, true,  false,
            true,  false, true, true,  true,  true, true, false};
    core::Tensor corres_select_bool(corres_select_bool_vec, {35},
                                    core::Dtype::Bool, device);
    std::vector<int64_t> corres_indices_vec{19, 5,  21, 3, 4,  18, 1,  15, 21,
                                            3,  6,  9,  8, 10, 16, 19, 7,  14,
                                            7,  10, 11, 9, 16, 16, 6,  11};
    core::Tensor corres_indices(corres_indices_vec, {26}, core::Dtype::Int64,
                                device);
    t::pipelines::registration::CorrespondenceSet corres =
            std::make_pair(corres_select_bool, corres_indices);

    t::pipelines::registration::TransformationEstimationPointToPlane
            estimation_p2plane;
    core::Tensor p2plane_transform = estimation_p2plane.ComputeTransformation(
            source_device, target_device, corres);
    t::geometry::PointCloud source_transformed_p2plane = source_device.Clone();
    source_transformed_p2plane.Transform(p2plane_transform.To(dtype));
    double p2plane_rmse_ = estimation_p2plane.ComputeRMSE(
            source_transformed_p2plane, target_device, corres);

    // Compare the new RMSE, after transformation.
    EXPECT_NEAR(p2plane_rmse_, 0.33768, 0.0005);
}

}  // namespace tests
}  // namespace open3d

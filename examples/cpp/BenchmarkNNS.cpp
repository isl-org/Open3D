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

// To run, from Open3D directory:
/* [CPU]: Open3D$ ./build/bin/examples/BenchmarkNNS CPU:0 \
                    examples/test_data/ICP/cloud_bin_0.pcd \
                    examples/test_data/ICP/cloud_bin_1.pcd
  [CUDA]: Open3D$ ./build/bin/examples/BenchmarkNNS CUDA:0 \
                    examples/test_data/ICP/cloud_bin_0.pcd \
                    examples/test_data/ICP/cloud_bin_1.pcd
*/

#include "open3d/Open3D.h"
using namespace open3d;

// Parameters to adjust according to the test pointcloud.
double voxel_downsample_factor = 1.0;
double max_correspondence_dist = 0.2;
int iterations = 5;

// Prepare source and target pointcloud on device, from a single input source.
inline void PrepareInput(t::geometry::PointCloud &input,
                         core::Tensor &transformation_,
                         t::geometry::PointCloud &target_device,
                         t::geometry::PointCloud &source_device,
                         core::Device device,
                         core::Dtype dtype);

int main(int argc, char *argv[]) {
    // Argument 1: Device: 'CPU:0' for CPU, 'CUDA:0' for GPU
    // Argument 2: Path to the test PointCloud
    core::Device device = core::Device(argv[1]);
    core::Dtype dtype = core::Dtype::Float32;

    // t::io::ReadPointCloud, changes the device to CPU and DType to Float64
    t::geometry::PointCloud input_;
    // t::geometry::PointCloud target(device);
    t::io::ReadPointCloud(argv[2], input_, {"auto", false, false, true});
    utility::LogInfo(" Input Successful ");

    // Creating Tensor from manual transformation vector.
    // target pointcloud = source.Transform(transformation_);
    core::Tensor transformation_ =
            core::Tensor::Init<float>({{0.862, 0.011, -0.507, 0.5},
                                       {-0.139, 0.967, -0.215, 0.7},
                                       {0.487, 0.255, 0.835, -1.4},
                                       {0.0, 0.0, 0.0, 1.0}},
                                      core::Device("CPU:0"));

    t::geometry::PointCloud target_device(device);
    t::geometry::PointCloud source_device(device);
    PrepareInput(input_, transformation_, target_device, source_device, device,
                 dtype);
    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
    utility::LogInfo(" Processing Input on {} Success", device.ToString());

    open3d::core::nns::NearestNeighborSearch target_nns(
            target_device.GetPoints());
    bool check = target_nns.HybridIndex(max_correspondence_dist);
    if (!check) {
        utility::LogError("Index Failed");
    }

    utility::LogInfo(" Source PointCloud size {}, Target PointCloud size {}",
                     source_device.GetPoints().GetShape()[0],
                     target_device.GetPoints().GetShape()[0]);

    double avg_ = 0, max_ = 0, min_ = 0;
    for (int i = 0; i < iterations; i++) {
        utility::Timer hybrid_time;

        // --- TIMER START
        hybrid_time.Start();
        auto result_nns = target_nns.HybridSearch(source_device.GetPoints(),
                                                  max_correspondence_dist, 1);
        hybrid_time.Stop();
        // --- TIMER STOP

        // To get number of correspondence
        auto correspondence_set_ =
                result_nns.first
                        .IndexGet({result_nns.first.Ne(-1).Reshape({-1})})
                        .Reshape({-1});
        utility::LogInfo(" [Tensor] HYBRID SEARCH TOOK {}, Correspondences: {}",
                         hybrid_time.GetDuration(),
                         correspondence_set_.GetShape()[0]);

        auto time = hybrid_time.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)iterations;

    utility::LogInfo(" Average Time: {}, Max {}, Min {} ", avg_, max_, min_);

    return 0;
}

inline void PrepareInput(t::geometry::PointCloud &input,
                         core::Tensor &transformation_,
                         t::geometry::PointCloud &target_device,
                         t::geometry::PointCloud &source_device,
                         core::Device device,
                         core::Dtype dtype) {
    // geometry::PointCloud legacy_s = source_.ToLegacyPointCloud();
    geometry::PointCloud legacy_t = input.ToLegacyPointCloud();

    // legacy_s.VoxelDownSample(voxel_downsample_factor);
    legacy_t.VoxelDownSample(voxel_downsample_factor);
    utility::LogInfo(" Downsampling Successful ");

    // legacy_t.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(),
    // false); utility::LogInfo(" Normal Estimation Successful ");

    t::geometry::PointCloud source =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    t::geometry::PointCloud target =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    target = target.Transform(transformation_);
    utility::LogInfo(" Target transformation Successful ");

    core::Tensor source_points =
            source.GetPoints().To(device, dtype, /*copy=*/true);
    source_device.SetPoints(source_points);
    utility::LogInfo(" Creating Source Pointcloud on device Successful ");

    core::Tensor target_points =
            target.GetPoints().To(device, dtype, /*copy=*/true);
    // core::Tensor target_normals =
    //         target.GetPointNormals().To(device, dtype, /*copy=*/true);
    target_device.SetPoints(target_points);
    // target_device.SetPointNormals(target_normals);
    utility::LogInfo(" Creating Target Pointcloud on device Successful ");
}

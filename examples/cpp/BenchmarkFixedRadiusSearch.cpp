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
/* Open3D$ ./build/bin/examples/BenchmarkFixedRadiusSearch
 */

#include "open3d/Open3D.h"
using namespace open3d;

// Parameters to adjust according to the test pointcloud.
double voxel_downsample_factor = 0.01;
double max_correspondence_dist = 0.1;
int iterations = 1;

// Prepare source and target pointcloud on device, from a single input source.
inline void PrepareInput(t::geometry::PointCloud &target,
                         t::geometry::PointCloud &source,
                         t::geometry::PointCloud &target_device,
                         t::geometry::PointCloud &source_device,
                         core::Device device,
                         core::Dtype dtype);

int main(int argc, char *argv[]) {
    core::Device device = core::Device(argv[1]);
    core::Dtype dtype = core::Dtype::Float32;

    // t::io::ReadPointCloud, changes the device to CPU and DType to Float64
    t::geometry::PointCloud source_;
    t::geometry::PointCloud target_;

    t::io::ReadPointCloud(argv[2], source_, {"auto", false, false, true});
    t::io::ReadPointCloud(argv[2], target_, {"auto", false, false, true});
    utility::LogInfo(" Input Successful ");

    // Creating Tensor from manual transformation vector.
    core::Tensor transformation_ = core::Tensor::Init<double>(
            {{0.99500417, -0.09933467, 0.00996671, 0.},
             {0.09983342, 0.99003329, -0.09933467, 0.},
             {0., 0.09983342, 0.99500417, 0.},
             {0., 0., 0., 1.}},
            core::Device("CPU:0"));
    source_ = source_.Transform(transformation_);

    t::geometry::PointCloud target_device(device);
    t::geometry::PointCloud source_device(device);
    PrepareInput(target_, source_, target_device, source_device, device, dtype);

    utility::LogInfo(" Processing Input on {} Success", device.ToString());
    bool sort = argc > 3 && strcmp(argv[3], "sort") == 0;

    open3d::core::nns::NearestNeighborSearch target_nns(
            target_device.GetPoints());
    bool check = target_nns.FixedRadiusIndex(max_correspondence_dist);
    if (!check) {
        utility::LogError("Index Failed");
    }

    utility::LogInfo(" Source PointCloud size {}, Target PointCloud size {}",
                     source_device.GetPoints().GetShape()[0],
                     target_device.GetPoints().GetShape()[0]);

    double avg_ = 0, max_ = 0, min_ = INT_MAX;
    for (int i = 0; i < iterations; i++) {
        utility::Timer radius_time;

        // --- TIMER START
        radius_time.Start();
        auto result_nns = target_nns.FixedRadiusSearch(
                source_device.GetPoints(), max_correspondence_dist,
                /* sort */ sort);
        radius_time.Stop();
        // --- TIMER STOP

        // To get number of correspondence
        auto correspondence_set_ = std::get<0>(result_nns);
        utility::LogInfo(
                " [Tensor] FIXEDRADIUS SEARCH TOOK {}, Correspondences: {}",
                radius_time.GetDuration(), correspondence_set_.GetShape()[0]);

        auto time = radius_time.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);

        core::Tensor neighbors_row_splits = std::get<2>(result_nns);
        auto size = neighbors_row_splits.GetShape()[0];
        utility::LogInfo(" size: {}", size);
        core::Tensor num_neighbors = neighbors_row_splits.Slice(0, 1, size) -
                                     neighbors_row_splits.Slice(0, 0, size - 1);
        utility::LogInfo(" Max neighbors: {}",
                         num_neighbors.Max({0}).Item<int64_t>());
    }
    avg_ = avg_ / (double)iterations;

    utility::LogInfo(" Average Time: {}, Max {}, Min {} ", avg_, max_, min_);

    return 0;
}

inline void PrepareInput(t::geometry::PointCloud &target_,
                         t::geometry::PointCloud &source_,
                         t::geometry::PointCloud &target_device,
                         t::geometry::PointCloud &source_device,
                         core::Device device,
                         core::Dtype dtype) {
    geometry::PointCloud legacy_s = source_.ToLegacyPointCloud();
    geometry::PointCloud legacy_t = target_.ToLegacyPointCloud();

    //     legacy_s = *legacy_s.VoxelDownSample(voxel_downsample_factor);
    //     legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);
    //     utility::LogInfo(" Downsampling Successful ");

    t::geometry::PointCloud source =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_s);

    t::geometry::PointCloud target =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    core::Tensor source_points =
            source.GetPoints().To(device, dtype, /*copy=*/true);
    source_device.SetPoints(source_points);
    utility::LogInfo(" Creating Source Pointcloud on device Successful ");

    core::Tensor target_points =
            target.GetPoints().To(device, dtype, /*copy=*/true);
    target_device.SetPoints(target_points);
    utility::LogInfo(" Creating Target Pointcloud on device Successful ");
}

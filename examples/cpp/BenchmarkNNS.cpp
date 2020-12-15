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

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"
// Headers files are moved to Open3d.h.in, in unmerged PR
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/utility/Timer.h"

using namespace open3d;

int main(int argc, char *argv[]) {
    core::Device device = core::Device(argv[1]);
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source_(device);
    t::io::ReadPointCloud(argv[2], source_, {"auto", false, false, true});

    t::geometry::PointCloud target_(device);
    t::io::ReadPointCloud(argv[3], target_, {"auto", false, false, true});

    core::Tensor source_points = source_.GetPoints().To(dtype).Copy(device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);
    core::Tensor target_points = target_.GetPoints().To(dtype).Copy(device);
    core::Tensor target_normals =
            target_.GetPointNormals().To(dtype).Copy(device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);

    double max_correspondence_dist = 0.02;
    utility::Timer eval_timer;
    double avg_ = 0.0;
    double max_ = 0.0;
    double min_ = 1000000.0;
    int itr = 10;

    open3d::core::nns::NearestNeighborSearch target_nns(
            target_device.GetPoints());
    bool check = target_nns.HybridIndex();
    if (!check) {
        utility::LogError("Index Failed");
    }

    max_correspondence_dist = max_correspondence_dist * max_correspondence_dist;

    utility::LogInfo(" Source PointCloud size {}, Target PointCloud size {}",
                     source_device.GetPoints().GetShape()[0],
                     target_device.GetPoints().GetShape()[0]);

    for (int i = 0; i < itr; i++) {
        core::Tensor transformation = init_trans;
        utility::Timer hybrid_time;

        // --- TIMER START
        hybrid_time.Start();
        auto result_nns = target_nns.HybridSearch(source_device.GetPoints(),
                                                  max_correspondence_dist, 1);
        hybrid_time.Stop();
        // --- TIMER STOP

        // To get number of correspondence
        auto correspondence_set_ =
            result_nns.first.IndexGet({result_nns.first.Ne(-1).Reshape({-1})})
                    .Reshape({-1});
        utility::LogInfo(" [Tensor] HYBRID SEARCH TOOK {}, Correspondences: {}",
                         hybrid_time.GetDuration(), correspondence_set_.GetShape()[0]);

        auto time = hybrid_time.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)itr;

    utility::LogInfo(" Average Time: {}, Max {}, Min {} ", avg_, max_, min_);

    return 0;
}

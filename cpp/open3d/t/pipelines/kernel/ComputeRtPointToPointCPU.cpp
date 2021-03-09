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

#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/t/pipelines/kernel/ComputeRtPointToPointImp.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputeRtPointToPointCPU(const float *src_pcd_ptr,
                              const float *tar_pcd_ptr,
                              const int64_t *corres_first,
                              const int64_t *corres_second,
                              const int n,
                              core::Tensor &R,
                              core::Tensor &t,
                              const core::Dtype dtype,
                              const core::Device device) {
    // Float64 is used for solving for higher precision.
    core::Dtype solve_dtype = core::Dtype::Float32;

    // sxy: {n, 9} Stores local sum for Sxy stacked vertically
    core::Tensor mean_s = core::Tensor::Empty({1, 3}, solve_dtype, device);
    float *mean_s_ptr = mean_s.GetDataPtr<float>();

    core::Tensor mean_t = core::Tensor::Empty({1, 3}, solve_dtype, device);
    float *mean_t_ptr = mean_t.GetDataPtr<float>();

    // sxy: {n, 9} Stores local sum for Sxy stacked vertically
    core::Tensor sxy = core::Tensor::Zeros({3, 3}, solve_dtype, device);
    float *sxy_ptr = sxy.GetDataPtr<float>();

    float num_correspondences = static_cast<float>(n);

    std::vector<double> mean_1x6(6, 0.0);
#ifdef _WIN32
    // Identity element for running_total reduction variable: zeros_6.
    std::vector<double> zeros_6(6, 0.0);
    // For TBB reduction, mean_ is a reduction variable of type vector<double>.
    mean_1x6 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_6,
            [&](tbb::blocked_range<int> r, std::vector<double> mean_) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    // For OpenMP reduction, mean_ is a double pointer to mean_1x6.
    double *mean_ = mean_1x6.data();
#pragma omp parallel for reduction(+ : mean_[:6])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
#endif
                    for (int i = 0; i < 3; i++) {
                        mean_[i] +=
                                src_pcd_ptr[3 * corres_first[workload_idx] + i];
                        mean_[i + 3] +=
                                tar_pcd_ptr[3 * corres_second[workload_idx] +
                                            i];
                    }
                }
#ifdef _WIN32
                return mean_;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(6);
                for (int j = 0; j < 6; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif

    for (int i = 0; i < 6; i++) {
        mean_[i] = mean_[i] / num_correspondences;
    }

    std::vector<double> sxy_1x9(9, 0.0);
#ifdef _WIN32
    // Identity element for running_total reduction variable: zeros_6.
    std::vector<double> zeros_9(9, 0.0);
    // For TBB reduction, sxy_1x9_ is a reduction variable of type
    // vector<double>.
    sxy_1x9 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_9,
            [&](tbb::blocked_range<int> r, std::vector<double> sxy_1x9_) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    // For OpenMP reduction, sxy_1x9_ is a double pointer to sxy_1x9.
    double *sxy_1x9_ = sxy_1x9.data();
#pragma omp parallel for reduction(+ : sxy_1x9_[:9])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
#endif
                    for (int i = 0; i < 9; i++) {
                        float s_ = src_pcd_ptr[3 * corres_first[workload_idx] +
                                               i % 3] -
                                   mean_1x6[i % 3];
                        float t_ = tar_pcd_ptr[3 * corres_second[workload_idx] +
                                               i / 3] -
                                   mean_1x6[3 + i / 3];
                        sxy_1x9_[i] += (t_) * (s_);
                    }
                }
#ifdef _WIN32
                return sxy_1x9_;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(9);
                for (int j = 0; j < 9; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif

    for (int i = 0, j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            sxy_ptr[j * 3 + k] = sxy_1x9[i++];
        }
        mean_s_ptr[j] = mean_1x6[j];
        mean_t_ptr[j] = mean_1x6[j + 3];
    }

    core::Tensor U, D, VT;
    std::tie(U, D, VT) = sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (U.Det() * (VT.T()).Det() < 0) {
        S[-1][-1] = -1;
    }

    R = U.Matmul(S.Matmul(VT));
    t = mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1});
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

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

#include "open3d/core/ParallelFor.h"
#include "open3d/core/linalg/kernel/Matrix.h"
#include "open3d/t/pipelines/kernel/Feature.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

#ifndef __CUDACC__
using std::max;
using std::min;
#endif

template <typename scalar_t>
OPEN3D_HOST_DEVICE void ComputePairFeature(const scalar_t *p1,
                                           const scalar_t *n1,
                                           const scalar_t *p2,
                                           const scalar_t *n2,
                                           scalar_t *feature) {
    scalar_t dp2p1[3], n1_copy[3], n2_copy[3];
    dp2p1[0] = p2[0] - p1[0];
    dp2p1[1] = p2[1] - p1[1];
    dp2p1[2] = p2[2] - p1[2];
    feature[3] = sqrt(dp2p1[0] * dp2p1[0] + dp2p1[1] * dp2p1[1] +
                      dp2p1[2] * dp2p1[2]);
    if (feature[3] == 0) {
        feature[0] = 0;
        feature[1] = 0;
        feature[2] = 0;
        feature[3] = 0;
        return;
    }

    const scalar_t angle1 =
            core::linalg::kernel::dot_3x1(n1, dp2p1) / feature[3];
    const scalar_t angle2 =
            core::linalg::kernel::dot_3x1(n2, dp2p1) / feature[4];
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy[0] = n2[0];
        n1_copy[1] = n2[1];
        n1_copy[2] = n2[2];
        n2_copy[0] = n1[0];
        n2_copy[1] = n1[1];
        n2_copy[2] = n1[2];
        dp2p1[0] *= -1;
        dp2p1[1] *= -1;
        dp2p1[2] *= -1;
        feature[2] = -angle2;
    } else {
        n1_copy[0] = n1[0];
        n1_copy[1] = n1[1];
        n1_copy[2] = n1[2];
        n2_copy[0] = n2[0];
        n2_copy[1] = n2[1];
        n2_copy[2] = n2[2];
        feature[2] = angle1;
    }

    scalar_t v[3];
    core::linalg::kernel::cross_3x1(dp2p1, n1_copy, v);
    const scalar_t v_norm = sqrt(core::linalg::kernel::dot_3x1(v, v));
    if (v_norm == 0.0) {
        feature[0] = 0.0;
        feature[1] = 0.0;
        feature[2] = 0.0;
        feature[3] = 0.0;
        return;
    }
    v[0] /= v_norm;
    v[1] /= v_norm;
    v[2] /= v_norm;
    scalar_t w[3];
    feature[1] = core::linalg::kernel::dot_3x1(v, n2_copy);
    feature[0] = atan2(core::linalg::kernel::dot_3x1(w, n2_copy),
                       core::linalg::kernel::dot_3x1(n1_copy, n2_copy));
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void UpdateSPFHFeature(const scalar_t *feature,
                                          int64_t idx,
                                          scalar_t hist_incr,
                                          scalar_t *spfh) {
    int h_index =
            static_cast<int>(floor(11 * (feature[0] + M_PI) / (2.0 * M_PI)));
    h_index = max(0, h_index);
    if (h_index >= 11) {
        h_index = 10;
    }
    spfh[idx * 33 + h_index] += hist_incr;

    h_index = static_cast<int>(floor(11 * (feature[1] + 1.0) * 0.5));
    h_index = max(0, h_index);
    if (h_index >= 11) {
        h_index = 10;
    }
    spfh[idx * 33 + h_index + 11] += hist_incr;

    h_index = static_cast<int>(floor(11 * (feature[2] + 1.0) * 0.5));
    h_index = max(0, h_index);
    if (h_index >= 11) {
        h_index = 10;
    }
    spfh[idx * 33 + h_index + 22] += hist_incr;
}

#if defined(__CUDACC__)
void ComputeSPFHFeatureCUDA
#else
void ComputeSPFHFeatureCPU
#endif
        (const core::Tensor &points,
         const core::Tensor &normals,
         const core::Tensor &indices,
         const core::Tensor &counts,
         core::Tensor &spfhs) {
    const core::Dtype dtype = points.GetDtype();
    const int64_t n = points.GetLength();
    const int64_t num_nn = indices.GetShape()[1];

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t *normals_ptr = normals.GetDataPtr<scalar_t>();
        const int32_t *indices_ptr = indices.GetDataPtr<int32_t>();
        const int64_t *counts_ptr = counts.GetDataPtr<int64_t>();
        scalar_t *spfhs_ptr = spfhs.GetDataPtr<scalar_t>();

        core::ParallelFor(points.GetDevice(), n,
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              int64_t idx = 3 * workload_idx;
                          });
    });
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

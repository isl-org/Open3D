// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
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

    scalar_t angle1 = core::linalg::kernel::dot_3x1(n1, dp2p1) / feature[3];
    scalar_t angle2 = core::linalg::kernel::dot_3x1(n2, dp2p1) / feature[3];
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
    const scalar_t v_norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
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
    core::linalg::kernel::cross_3x1(n1_copy, v, w);
    feature[1] = core::linalg::kernel::dot_3x1(v, n2_copy);
    feature[0] = atan2(core::linalg::kernel::dot_3x1(w, n2_copy),
                       core::linalg::kernel::dot_3x1(n1_copy, n2_copy));
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE void UpdateSPFHFeature(const scalar_t *feature,
                                          int64_t idx,
                                          scalar_t hist_incr,
                                          scalar_t *spfh) {
    int h_index1 =
            static_cast<int>(floor(11 * (feature[0] + M_PI) / (2.0 * M_PI)));
    h_index1 = h_index1 >= 11 ? 10 : max(0, h_index1);

    int h_index2 = static_cast<int>(floor(11 * (feature[1] + 1.0) * 0.5));
    h_index2 = h_index2 >= 11 ? 10 : max(0, h_index2);

    int h_index3 = static_cast<int>(floor(11 * (feature[2] + 1.0) * 0.5));
    h_index3 = h_index3 >= 11 ? 10 : max(0, h_index3);

    spfh[idx * 33 + h_index1] += hist_incr;
    spfh[idx * 33 + h_index2 + 11] += hist_incr;
    spfh[idx * 33 + h_index3 + 22] += hist_incr;
}

#if defined(__CUDACC__)
void ComputeFPFHFeatureCUDA
#else
void ComputeFPFHFeatureCPU
#endif
        (const core::Tensor &points,
         const core::Tensor &normals,
         const core::Tensor &indices,
         const core::Tensor &distance2,
         const core::Tensor &counts,
         core::Tensor &fpfhs,
         const utility::optional<core::Tensor> &mask,
         const utility::optional<core::Tensor> &map_info_idx_to_point_idx) {
    const core::Dtype dtype = points.GetDtype();
    const core::Device device = points.GetDevice();
    const int64_t n_points = points.GetLength();

    const bool filter_fpfh =
            mask.has_value() && map_info_idx_to_point_idx.has_value();
    if (mask.has_value() ^ map_info_idx_to_point_idx.has_value()) {
        utility::LogError(
                "Parameters mask and map_info_idx_to_point_idx must "
                "either be both provided or both not provided.");
    }
    if (filter_fpfh) {
        if (mask.value().GetShape()[0] != n_points) {
            utility::LogError(
                    "Parameter mask was provided, but its size {:d} should"
                    "be equal to the number of points {:d}.",
                    (int)mask.value().GetShape()[0], n_points);
        }
        if (map_info_idx_to_point_idx.value().GetShape()[0] !=
            counts.GetShape()[0] - (indices.GetShape().size() == 1 ? 1 : 0)) {
            utility::LogError(
                    "Parameter map_info_idx_to_point_idx was provided, "
                    "but its size"
                    "{:d} should be equal to the size of counts {:d}.",
                    (int)map_info_idx_to_point_idx.value().GetShape()[0],
                    (int)counts.GetShape()[0]);
        }
    }

    core::Tensor map_spfh_info_idx_to_point_idx =
            map_info_idx_to_point_idx.value_or(
                    core::Tensor::Empty({0}, core::Int64, device));

    const core::Tensor map_fpfh_idx_to_point_idx =
            filter_fpfh ? mask.value().NonZero().GetItem(
                                  {core::TensorKey::Index(0)})
                        : core::Tensor::Empty({0}, core::Int64, device);

    const int32_t n_fpfh =
            filter_fpfh ? map_fpfh_idx_to_point_idx.GetLength() : n_points;
    const int32_t n_spfh =
            filter_fpfh ? map_spfh_info_idx_to_point_idx.GetLength() : n_points;

    core::Tensor spfhs =
            core::Tensor::Zeros({n_spfh, 33}, dtype, fpfhs.GetDevice());

    core::Tensor map_point_idx_to_spfh_idx;
    if (filter_fpfh) {
        map_point_idx_to_spfh_idx = core::Tensor::Full(
                {n_points}, -1, core::Int64, fpfhs.GetDevice());
        map_point_idx_to_spfh_idx.IndexSet(
                {map_spfh_info_idx_to_point_idx},
                core::Tensor::Arange(0, n_spfh, 1, core::Int64,
                                     fpfhs.GetDevice()));
    } else {
        map_point_idx_to_spfh_idx =
                core::Tensor::Empty({0}, core::Int64, fpfhs.GetDevice());
    }

    // Check the nns type (knn = hybrid = false, radius = true).
    // The nns radius search mode will resulting a prefix sum 1D tensor.
    bool is_radius_search;
    int nn_size = 0;
    if (indices.GetShape().size() == 1) {
        is_radius_search = true;
    } else {
        is_radius_search = false;
        nn_size = indices.GetShape()[1];
    }

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t *normals_ptr = normals.GetDataPtr<scalar_t>();
        const int32_t *indices_ptr = indices.GetDataPtr<int32_t>();
        const scalar_t *distance2_ptr = distance2.GetDataPtr<scalar_t>();
        const int32_t *counts_ptr = counts.GetDataPtr<int32_t>();
        scalar_t *spfhs_ptr = spfhs.GetDataPtr<scalar_t>();
        scalar_t *fpfhs_ptr = fpfhs.GetDataPtr<scalar_t>();
        const int64_t *map_spfh_info_idx_to_point_idx_ptr =
                map_spfh_info_idx_to_point_idx.GetDataPtr<int64_t>();
        const int64_t *map_fpfh_idx_to_point_idx_ptr =
                map_fpfh_idx_to_point_idx.GetDataPtr<int64_t>();
        const int64_t *map_point_idx_to_spfh_idx_ptr =
                map_point_idx_to_spfh_idx.GetDataPtr<int64_t>();

        // Compute SPFH features for the points.
        core::ParallelFor(
                points.GetDevice(), n_spfh,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t workload_point_idx =
                            filter_fpfh ? map_spfh_info_idx_to_point_idx_ptr
                                                  [workload_idx]
                                        : workload_idx;
                    int64_t idx = 3 * workload_point_idx;
                    const scalar_t *point = points_ptr + idx;
                    const scalar_t *normal = normals_ptr + idx;

                    const int indice_size =
                            is_radius_search ? (counts_ptr[workload_idx + 1] -
                                                counts_ptr[workload_idx])
                                             : counts_ptr[workload_idx];

                    if (indice_size > 1) {
                        const scalar_t hist_incr =
                                100.0 / static_cast<scalar_t>(indice_size - 1);
                        for (int i = 1; i < indice_size; i++) {
                            const int point_idx =
                                    is_radius_search
                                            ? indices_ptr
                                                      [i +
                                                       counts_ptr[workload_idx]]
                                            : indices_ptr[workload_idx *
                                                                  nn_size +
                                                          i];

                            const scalar_t *point_ref =
                                    points_ptr + 3 * point_idx;
                            const scalar_t *normal_ref =
                                    normals_ptr + 3 * point_idx;
                            scalar_t fea[4] = {0};
                            ComputePairFeature<scalar_t>(
                                    point, normal, point_ref, normal_ref, fea);
                            UpdateSPFHFeature<scalar_t>(fea, workload_idx,
                                                        hist_incr, spfhs_ptr);
                        }
                    }
                });

        // Compute FPFH features for the points.
        core::ParallelFor(
                points.GetDevice(), n_fpfh,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t workload_spfh_idx =
                            filter_fpfh ? map_point_idx_to_spfh_idx_ptr
                                                  [map_fpfh_idx_to_point_idx_ptr
                                                           [workload_idx]]
                                        : workload_idx;
                    const int indice_size =
                            is_radius_search
                                    ? (counts_ptr[workload_spfh_idx + 1] -
                                       counts_ptr[workload_spfh_idx])
                                    : counts_ptr[workload_spfh_idx];
                    if (indice_size > 1) {
                        scalar_t sum[3] = {0.0, 0.0, 0.0};
                        for (int i = 1; i < indice_size; i++) {
                            const int idx =
                                    is_radius_search
                                            ? i + counts_ptr[workload_spfh_idx]
                                            : workload_spfh_idx * nn_size + i;
                            const scalar_t dist = distance2_ptr[idx];
                            if (dist == 0.0) continue;
                            const int32_t spfh_idx =
                                    filter_fpfh ? map_point_idx_to_spfh_idx_ptr
                                                          [indices_ptr[idx]]
                                                : indices_ptr[idx];
                            for (int j = 0; j < 33; j++) {
                                const scalar_t val =
                                        spfhs_ptr[spfh_idx * 33 + j] / dist;
                                sum[j / 11] += val;
                                fpfhs_ptr[workload_idx * 33 + j] += val;
                            }
                        }
                        for (int j = 0; j < 3; j++) {
                            sum[j] = sum[j] != 0.0 ? 100.0 / sum[j] : 0.0;
                        }
                        for (int j = 0; j < 33; j++) {
                            fpfhs_ptr[workload_idx * 33 + j] *= sum[j / 11];
                            fpfhs_ptr[workload_idx * 33 + j] +=
                                    spfhs_ptr[workload_spfh_idx * 33 + j];
                        }
                    }
                });
    });
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

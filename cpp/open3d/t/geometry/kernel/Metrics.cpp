// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"
namespace open3d {
namespace t {
namespace geometry {

core::Tensor ComputeMetricsCommon(core::Tensor distance12,
                                  core::Tensor distance21,
                                  std::vector<Metric> metrics,
                                  MetricParameters params) {
    int n_metrics = metrics.size();
    if (std::find(metrics.begin(), metrics.end(), Metric::FScore) !=
        metrics.end()) {
        n_metrics += params.fscore_radius.size() - 1;
    }
    core::Tensor metric_values({n_metrics}, core::Float32,
                               distance12.GetDevice());
    float metric_val;

    int idx = 0;
    for (Metric metric : metrics) {
        switch (metric) {
            case Metric::ChamferDistance:
                metric_val = distance21.Reshape({-1}).Mean({-1}).Item<float>() +
                             distance12.Reshape({-1}).Mean({-1}).Item<float>();
                metric_values[idx++] = metric_val;
                break;
            case Metric::HausdorffDistance:
                metric_val = std::max(
                        distance12.Reshape({-1}).Max({-1}).Item<float>(),
                        distance21.Reshape({-1}).Max({-1}).Item<float>());
                metric_values[idx++] = metric_val;
                break;
            case Metric::FScore:
                float *p_distance12 = distance12.GetDataPtr<float>(),
                      *p_distance21 = distance21.GetDataPtr<float>();
                for (float radius : params.fscore_radius) {
                    // Workaround since we don't have Tensor::CountNonZeros
                    float precision = 0., recall = 0.;
                    for (size_t i = 0;
                         i < static_cast<size_t>(distance12.NumElements()); ++i)
                        precision += p_distance12[i] < radius;
                    precision *= 100. / distance12.NumElements();
                    for (size_t i = 0;
                         i < static_cast<size_t>(distance21.NumElements()); ++i)
                        recall += p_distance21[i] < radius;
                    recall *= 100. / distance21.NumElements();
                    float fscore = 0.0;
                    if (precision + recall > 0) {
                        fscore = 2 * precision * recall / (precision + recall);
                    }
                    metric_values[idx++] = fscore;
                }
                break;
        }
    }
    return metric_values;
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d

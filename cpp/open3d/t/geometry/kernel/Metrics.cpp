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

std::vector<float> OPEN3D_LOCAL
ComputeDistanceCommon(core::Tensor distance12,
                      core::Tensor distance21,
                      std::initializer_list<Metric> metrics,
                      MetricParameters params) {
    std::vector<float> metric_values;
    float metric_val;

    for (Metric metric : metrics) {
        switch (metric) {
            case Metric::ChamferDistance:
                metric_val = 0.5 * (distance21.Mean({-1}).Item<float>() +
                                    distance12.Mean({-1}).Item<float>());
                metric_values.push_back(metric_val);
                break;
            case Metric::FScore:
                float *p_distance12 = distance12.GetDataPtr<float>(),
                      *p_distance21 = distance21.GetDataPtr<float>();
                for (float radius : params.fscore_radius) {
                    // Workaround since we don't have Tensor::CountNonZeros
                    float precision = 0., recall = 0.;
                    for (size_t i = 0;
                         i < static_cast<size_t>(distance12.GetLength()); ++i)
                        precision += p_distance12[i] < radius;
                    precision *= 100. / distance12.GetLength();
                    for (size_t i = 0;
                         i < static_cast<size_t>(distance21.GetLength()); ++i)
                        recall += p_distance21[i] < radius;
                    recall *= 100. / distance21.GetLength();
                    float fscore = 0.0;
                    if (precision + recall > 0) {
                        fscore = 2 * precision * recall / (precision + recall);
                    }
                    metric_values.push_back(fscore);
                }
                break;
        }
    }
    return metric_values;
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d

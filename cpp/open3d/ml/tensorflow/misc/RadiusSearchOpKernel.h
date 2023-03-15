// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../TensorFlowHelper.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace radius_search_opkernel {

class RadiusSearchOpKernel : public tensorflow::OpKernel {
public:
    explicit RadiusSearchOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace open3d::core::nns;
        using namespace tensorflow;
        std::string metric_str;
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("metric", &metric_str));
        if (metric_str == "L1")
            metric = L1;
        else
            metric = L2;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("ignore_query_point",
                                             &ignore_query_point));

        OP_REQUIRES_OK(construction, construction->GetAttr("return_distances",
                                                           &return_distances));
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("normalize_distances",
                                             &normalize_distances));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& points = context->input(0);
        const Tensor& queries = context->input(1);
        const Tensor& radii = context->input(2);
        const Tensor& points_row_splits = context->input(3);
        const Tensor& queries_row_splits = context->input(4);
        {
            using namespace open3d::ml::op_util;

            Dim num_points("num_points");
            Dim num_queries("num_queries");
            Dim batch_size("batch_size");
            CHECK_SHAPE(context, points, num_points, 3);
            CHECK_SHAPE(context, queries, num_queries, 3);
            CHECK_SHAPE(context, radii, num_queries);
            CHECK_SHAPE(context, points_row_splits, batch_size + 1);
            CHECK_SHAPE(context, queries_row_splits, batch_size + 1);
        }

        Tensor* query_neighbors_row_splits = 0;
        TensorShape query_neighbors_row_splits_shape(
                {queries.shape().dim_size(0) + 1});
        OP_REQUIRES_OK(context, context->allocate_output(
                                        1, query_neighbors_row_splits_shape,
                                        &query_neighbors_row_splits));

        Kernel(context, points, queries, radii, points_row_splits,
               queries_row_splits, *query_neighbors_row_splits);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& queries,
                        const tensorflow::Tensor& radius,
                        const tensorflow::Tensor& points_row_splits,
                        const tensorflow::Tensor& queries_row_splits,
                        tensorflow::Tensor& query_neighbors_row_splits) = 0;

protected:
    open3d::core::nns::Metric metric;
    bool ignore_query_point;
    bool return_distances;
    bool normalize_distances;
};

}  // namespace radius_search_opkernel
/// @endcond

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../TensorFlowHelper.h"
#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace fixed_radius_search_opkernel {

// class for the allocator object
template <class T, class TIndex>
class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocIndices(TIndex** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));

        auto flat_tensor = tensor->flat<TIndex>();
        *ptr = (TIndex*)flat_tensor.data();
    }

    void AllocDistances(T** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(2, shape, &tensor));
        auto flat_tensor = tensor->flat<T>();
        *ptr = flat_tensor.data();
    }

private:
    tensorflow::OpKernelContext* context;
};

class FixedRadiusSearchOpKernel : public tensorflow::OpKernel {
public:
    explicit FixedRadiusSearchOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;
        using namespace open3d::core::nns;

        std::string metric_str;
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("metric", &metric_str));
        if (metric_str == "L1")
            metric = L1;
        else if (metric_str == "L2")
            metric = L2;
        else
            metric = Linf;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("ignore_query_point",
                                             &ignore_query_point));

        OP_REQUIRES_OK(construction, construction->GetAttr("return_distances",
                                                           &return_distances));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& points = context->input(0);
        const Tensor& queries = context->input(1);

        const Tensor& radius = context->input(2);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(radius.shape()),
                    errors::InvalidArgument("radius must be scalar, got shape ",
                                            radius.shape().DebugString()));

        const Tensor& points_row_splits = context->input(3);
        const Tensor& queries_row_splits = context->input(4);

        const Tensor& hash_table_splits = context->input(5);
        const Tensor& hash_table_index = context->input(6);
        const Tensor& hash_table_cell_splits = context->input(7);

        {
            using namespace open3d::ml::op_util;

            Dim num_points("num_points");
            Dim num_queries("num_queries");
            Dim batch_size("batch_size");
            Dim num_cells("num_cells");
            CHECK_SHAPE(context, points, num_points, 3);
            CHECK_SHAPE(context, hash_table_index, num_points);
            CHECK_SHAPE(context, queries, num_queries, 3);
            CHECK_SHAPE(context, points_row_splits, batch_size + 1);
            CHECK_SHAPE(context, queries_row_splits, batch_size + 1);
            CHECK_SHAPE(context, hash_table_splits, batch_size + 1);
            CHECK_SHAPE(context, hash_table_cell_splits, num_cells + 1);
        }
        Tensor* query_neighbors_row_splits = 0;
        TensorShape query_neighbors_row_splits_shape(
                {queries.shape().dim_size(0) + 1});
        OP_REQUIRES_OK(context, context->allocate_output(
                                        1, query_neighbors_row_splits_shape,
                                        &query_neighbors_row_splits));

        Kernel(context, points, queries, radius, points_row_splits,
               queries_row_splits, hash_table_splits, hash_table_index,
               hash_table_cell_splits, *query_neighbors_row_splits);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& queries,
                        const tensorflow::Tensor& radius,
                        const tensorflow::Tensor& points_row_splits,
                        const tensorflow::Tensor& queries_row_splits,
                        const tensorflow::Tensor& hash_table_splits,
                        const tensorflow::Tensor& hash_table_index,
                        const tensorflow::Tensor& hash_table_cell_splits,
                        tensorflow::Tensor& query_neighbors_row_splits) = 0;

protected:
    open3d::core::nns::Metric metric;
    bool ignore_query_point;
    bool return_distances;
};
}  // namespace fixed_radius_search_opkernel
/// @endcond

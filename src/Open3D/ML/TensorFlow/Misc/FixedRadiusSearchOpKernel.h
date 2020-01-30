// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
#pragma once

#include "Open3D/ML/Misc/Detail/FixedRadiusSearch.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

// namespace for code that is common for all kernels
namespace fixed_radius_search_opkernel {

// class for the allocator object
template <class T>
class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocIndices(tensorflow::int32** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<int32>();
        *ptr = flat_tensor.data();
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
        using namespace open3d::ml::detail;

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

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("max_hash_table_size",
                                             &max_hash_table_size));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& points = context->input(0);
        OP_REQUIRES(context, points.shape().dims() == 2,
                    errors::InvalidArgument("points must be a rank 2 tensor"));

        const Tensor& queries = context->input(1);
        OP_REQUIRES(context, queries.shape().dims() == 2,
                    errors::InvalidArgument("queries must be a rank 2 tensor"));

        const Tensor& radius = context->input(2);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(radius.shape()),
                    errors::InvalidArgument("radius must be scalar, got shape ",
                                            radius.shape().DebugString()));

        const Tensor& hash_table_size_factor_tensor = context->input(3);
        OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(
                        hash_table_size_factor_tensor.shape()),
                errors::InvalidArgument(
                        "hash_table_size_factor must be scalar, got shape ",
                        hash_table_size_factor_tensor.shape().DebugString()));
        const double hash_table_size_factor =
                hash_table_size_factor_tensor.scalar<double>()();
        const size_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(
                        hash_table_size_factor * points.shape().dim_size(0), 1),
                max_hash_table_size);

        Tensor* query_neighbors_prefix_sum = 0;
        TensorShape query_neighbors_prefix_sum_shape(
                {queries.shape().dim_size(0)});
        OP_REQUIRES_OK(context, context->allocate_output(
                                        1, query_neighbors_prefix_sum_shape,
                                        &query_neighbors_prefix_sum));

        Tensor hash_table;
        TensorShape hash_table_shape({ssize_t(hash_table_size)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint32_t>::v(),
                                              hash_table_shape, &hash_table));

        Kernel(context, points, queries, radius, hash_table_size,
               *query_neighbors_prefix_sum);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& queries,
                        const tensorflow::Tensor& radius,
                        const size_t hash_table_size,
                        tensorflow::Tensor& query_neighbors_prefix_sum) = 0;

protected:
    open3d::ml::detail::Metric metric;
    bool ignore_query_point;
    bool return_distances;
    int max_hash_table_size;
};
}  // namespace fixed_radius_search_opkernel

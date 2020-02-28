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

#include "Open3D/ML/Misc/Detail/KnnSearch.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

// namespace for code that is common for all kernels
namespace knn_search_opkernel {

template <class T>
class OutputAllocator {
public:
    OutputAllocator(tensorflow::OpKernelContext* context) : context(context) {}

    void AllocIndices(int32_t** ptr, size_t num) {
        using namespace tensorflow;
        *ptr = nullptr;
        Tensor* tensor = 0;
        TensorShape shape({int64_t(num)});
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));
        auto flat_tensor = tensor->flat<int32>();
        static_assert(sizeof(int32) == sizeof(int32_t),
                      "int32 and int32_t not compatible");
        *ptr = (int32_t*)flat_tensor.data();
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

class KnnSearchOpKernel : public tensorflow::OpKernel {
public:
    explicit KnnSearchOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace open3d::ml::detail;
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

        const Tensor& k_tensor = context->input(2);
        const TensorShape k_shape(k_tensor.shape());
        const int k_rank = k_shape.dims();
        OP_REQUIRES(context, k_rank == 0,
                    errors::InvalidArgument("k must be a rank 0 tensor"));
        const int k = k_tensor.scalar<int32_t>()();

        Tensor* query_neighbors_row_splits = 0;
        TensorShape query_neighbors_row_splits_shape(
                {queries.shape().dim_size(0) + 1});
        OP_REQUIRES_OK(context, context->allocate_output(
                                        1, query_neighbors_row_splits_shape,
                                        &query_neighbors_row_splits));

        Kernel(context, points, queries, k, *query_neighbors_row_splits);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& queries,
                        const int k,
                        tensorflow::Tensor& query_neighbors_row_splits) = 0;

protected:
    open3d::ml::detail::Metric metric;
    bool ignore_query_point;
    bool return_distances;
};

}  // namespace knn_search_opkernel

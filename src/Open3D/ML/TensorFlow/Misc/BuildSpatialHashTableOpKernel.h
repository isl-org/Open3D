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

#include "../TensorFlowHelper.h"
#include "Open3D/ML/Misc/Detail/FixedRadiusSearch.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

class BuildSpatialHashTableOpKernel : public tensorflow::OpKernel {
public:
    explicit BuildSpatialHashTableOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;
        using namespace open3d::ml::detail;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("max_hash_table_size",
                                             &max_hash_table_size));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        using namespace open3d::ml::shape_checking;

        const Tensor& points = context->input(0);

        Dim num_points("num_points");
        CHECK_SHAPE(context, points, num_points, 3);

        const Tensor& radius = context->input(1);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(radius.shape()),
                    errors::InvalidArgument("radius must be scalar, got shape ",
                                            radius.shape().DebugString()));

        const Tensor& hash_table_size_factor_tensor = context->input(2);
        OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(
                        hash_table_size_factor_tensor.shape()),
                errors::InvalidArgument(
                        "hash_table_size_factor must be scalar, got shape ",
                        hash_table_size_factor_tensor.shape().DebugString()));
        const double hash_table_size_factor =
                hash_table_size_factor_tensor.scalar<double>()();
        const int64_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(
                        hash_table_size_factor * points.shape().dim_size(0), 1),
                max_hash_table_size);

        Tensor* hash_table_index = 0;
        TensorShape hash_table_index_shape({num_points.value()});
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, hash_table_index_shape,
                                                &hash_table_index));

        Tensor* hash_table_row_splits = 0;
        TensorShape hash_table_row_splits_shape({hash_table_size + 1});
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, hash_table_row_splits_shape,
                                                &hash_table_row_splits));

        Kernel(context, points, radius, *hash_table_index,
               *hash_table_row_splits);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& radius,
                        tensorflow::Tensor& hash_table_index,
                        tensorflow::Tensor& hash_table_row_splits) = 0;

protected:
    int max_hash_table_size;
};

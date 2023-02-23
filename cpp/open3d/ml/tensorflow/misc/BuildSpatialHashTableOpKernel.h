// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../TensorFlowHelper.h"
#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

class BuildSpatialHashTableOpKernel : public tensorflow::OpKernel {
public:
    explicit BuildSpatialHashTableOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("max_hash_table_size",
                                             &max_hash_table_size));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        using namespace open3d::ml::op_util;

        const Tensor& points = context->input(0);
        const Tensor& radius = context->input(1);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(radius.shape()),
                    errors::InvalidArgument("radius must be scalar, got shape ",
                                            radius.shape().DebugString()));

        const Tensor& points_row_splits = context->input(2);

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

        Dim num_points("num_points");
        Dim batch_size("batch_size");
        CHECK_SHAPE(context, points, num_points, 3);
        CHECK_SHAPE(context, points_row_splits, batch_size + 1);

        std::vector<uint32_t> hash_table_splits(batch_size.value() + 1, 0);
        for (int i = 0; i < batch_size.value(); ++i) {
            int64_t num_points_i = points_row_splits.flat<int64>()(i + 1) -
                                   points_row_splits.flat<int64>()(i);

            int64_t hash_table_size = std::min<int64_t>(
                    std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
                    max_hash_table_size);
            hash_table_splits[i + 1] = hash_table_splits[i] + hash_table_size;
        }

        Tensor* hash_table_index = 0;
        TensorShape hash_table_index_shape({num_points.value()});
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, hash_table_index_shape,
                                                &hash_table_index));

        Tensor* hash_table_cell_splits = 0;
        TensorShape hash_table_cell_splits_shape(
                {hash_table_splits.back() + 1});
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, hash_table_cell_splits_shape,
                                                &hash_table_cell_splits));

        Tensor* out_hash_table_splits = 0;
        TensorShape out_hash_table_splits_shape({batch_size.value() + 1});
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, out_hash_table_splits_shape,
                                                &out_hash_table_splits));
        for (size_t i = 0; i < hash_table_splits.size(); ++i) {
            out_hash_table_splits->flat<uint32_t>()(i) = hash_table_splits[i];
        }

        Kernel(context, points, radius, points_row_splits, hash_table_splits,
               *hash_table_index, *hash_table_cell_splits);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& points,
                        const tensorflow::Tensor& radius,
                        const tensorflow::Tensor& points_row_splits,
                        const std::vector<uint32_t>& hash_table_splits,
                        tensorflow::Tensor& hash_table_index,
                        tensorflow::Tensor& hash_table_cell_splits) = 0;

protected:
    int max_hash_table_size;
};

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/lib/core/errors.h>

#include "open3d/ml/tensorflow/TensorFlowHelper.h"

template <class TIndex>
class SparseConvOpKernel : public tensorflow::OpKernel {
public:
    explicit SparseConvOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("normalize", &normalize));

        OP_REQUIRES_OK(construction, construction->GetAttr("max_temp_mem_MB",
                                                           &max_temp_mem_MB));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        using namespace open3d::ml::op_util;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");
        const Tensor& filters = context->input(0);
        const Tensor& inp_features = context->input(1);
        const Tensor& inp_importance = context->input(2);
        const Tensor& neighbors_index = context->input(3);
        const Tensor& neighbors_kernel_index = context->input(4);
        const Tensor& neighbors_importance = context->input(5);
        const Tensor& neighbors_row_splits = context->input(6);

        Dim num_out("num_out");
        Dim num_inp("num_inp");
        Dim num_kernel_elements("num_kernel_elements");
        Dim in_channels("in_channels");
        Dim out_channels("out_channels");
        Dim num_neighbors("num_neighbors");

        CHECK_SHAPE_COMBINE_FIRST_DIMS(context, filters, num_kernel_elements,
                                       in_channels, out_channels);
        CHECK_SHAPE(context, neighbors_row_splits, num_out + 1);
        CHECK_SHAPE(context, inp_features, num_inp, in_channels);
        CHECK_SHAPE(context, inp_importance, 0 || num_inp);
        CHECK_SHAPE(context, neighbors_index, num_neighbors);
        CHECK_SHAPE(context, neighbors_kernel_index, num_neighbors);
        CHECK_SHAPE(context, neighbors_importance, 0 || num_neighbors);

        TensorShape out_features_shape({num_out.value(), out_channels.value()});
        Tensor* out_features = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_features_shape,
                                                         &out_features));

        std::vector<int> filter_dims;
        for (int i = 0; i < filters.dims(); ++i) {
            filter_dims.push_back(filters.dim_size(i));
        }

        bool point_importances = inp_importance.shape().dim_size(0) != 0;

        bool has_neighbors_importances =
                neighbors_importance.shape().dim_size(0) != 0;

        Kernel(context, filters, inp_features, inp_importance, neighbors_index,
               neighbors_kernel_index, neighbors_importance,
               neighbors_row_splits, filter_dims, point_importances,
               has_neighbors_importances, *out_features);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& filters,
                        const tensorflow::Tensor& inp_features,
                        const tensorflow::Tensor& inp_importance,
                        const tensorflow::Tensor& neighbors_index,
                        const tensorflow::Tensor& neighbors_kernel_index,
                        const tensorflow::Tensor& neighbors_importance,
                        const tensorflow::Tensor& neighbors_row_splits,
                        const std::vector<int>& filter_dims,
                        const bool point_importances,
                        const bool has_neighbors_importances,
                        tensorflow::Tensor& out_features) = 0;

public:
    bool normalize;
    int max_temp_mem_MB;
};

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

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

template <class TIndex>
class ContinuousConvBackpropFilterOpKernel : public tensorflow::OpKernel {
public:
    explicit ContinuousConvBackpropFilterOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;
        using namespace open3d::ml::impl;
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("align_corners", &align_corners));
        OP_REQUIRES_OK(construction,
                       construction->GetAttr("normalize", &normalize));

        std::string interpolation_str;
        OP_REQUIRES_OK(construction, construction->GetAttr("interpolation",
                                                           &interpolation_str));

        if (interpolation_str == "linear")
            interpolation = InterpolationMode::LINEAR;
        else if (interpolation_str == "linear_border")
            interpolation = InterpolationMode::LINEAR_BORDER;
        else
            interpolation = InterpolationMode::NEAREST_NEIGHBOR;

        std::string mapping_str;
        OP_REQUIRES_OK(construction, construction->GetAttr("coordinate_mapping",
                                                           &mapping_str));

        if (mapping_str == "ball_to_cube_radial")
            coordinate_mapping = CoordinateMapping::BALL_TO_CUBE_RADIAL;
        else if (mapping_str == "ball_to_cube_volume_preserving")
            coordinate_mapping =
                    CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING;
        else
            coordinate_mapping = CoordinateMapping::IDENTITY;

        OP_REQUIRES_OK(construction, construction->GetAttr("max_temp_mem_MB",
                                                           &max_temp_mem_MB));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");
        const Tensor& filter = context->input(0);

        const Tensor& out_positions = context->input(1);
        OP_REQUIRES(context,
                    out_positions.shape().dim_size(0) <=
                            std::numeric_limits<TIndex>::max(),
                    errors::InvalidArgument("Too many output points"));

        const Tensor& extents = context->input(2);
        OP_REQUIRES(context, extents.shape().dims() == 2,
                    errors::InvalidArgument("extents must be a rank 2 tensor"));
        OP_REQUIRES(context,
                    extents.shape().dim_size(0) ==
                                    out_positions.shape().dim_size(0) ||
                            extents.shape().dim_size(0) == 1,
                    errors::InvalidArgument("number of extents must match the "
                                            "number of out_positions or must "
                                            "be 1"));
        OP_REQUIRES(context,
                    extents.shape().dim_size(1) == 3 ||
                            extents.shape().dim_size(1) == 1,
                    errors::InvalidArgument(
                            "number of components for extents must be 3 or 1"));

        const Tensor& offset = context->input(3);
        OP_REQUIRES(context, offset.shape().dims() == 1,
                    errors::InvalidArgument("offset must be a rank 1 tensor"));
        OP_REQUIRES(context, offset.shape().dim_size(0) == 3,
                    errors::InvalidArgument("offset length must be 3"));

        const Tensor& inp_positions = context->input(4);
        OP_REQUIRES(context,
                    inp_positions.shape().dim_size(0) <=
                            std::numeric_limits<TIndex>::max(),
                    errors::InvalidArgument("Too many input points"));

        const Tensor& inp_features = context->input(5);

        const Tensor& inp_importance = context->input(6);

        const Tensor& neighbors_index = context->input(7);

        const Tensor& neighbors_importance = context->input(8);

        const Tensor& neighbors_row_splits = context->input(9);

        const Tensor& out_features_gradient = context->input(10);

        OP_REQUIRES(
                context,
                inp_positions.shape().dim_size(0) ==
                        inp_features.shape().dim_size(0),
                errors::InvalidArgument("first dim of inp_positions does not "
                                        "match the first dim of inp_features"));

        OP_REQUIRES(context,
                    inp_positions.shape().dim_size(0) ==
                                    inp_importance.shape().dim_size(0) ||
                            inp_importance.shape().dim_size(0) == 0,
                    errors::InvalidArgument("first dim of inp_positions does "
                                            "not match the first dim of "
                                            "inp_importance"));

        OP_REQUIRES(context,
                    neighbors_importance.shape().dim_size(0) ==
                                    neighbors_index.shape().dim_size(0) ||
                            neighbors_importance.shape().dim_size(0) == 0,
                    errors::InvalidArgument("first dim of neighbors_importance "
                                            "does not match the first dim of "
                                            "neighbors_index"));

        OP_REQUIRES(
                context,
                filter.shape().dim_size(3) == inp_features.shape().dim_size(1),
                errors::InvalidArgument("number of input channels in filter "
                                        "and inp_features does not match"));

        OP_REQUIRES(context,
                    out_features_gradient.shape().dim_size(0) ==
                            out_positions.shape().dim_size(0),
                    errors::InvalidArgument("first dim of out_positions, does "
                                            "not match the first dim of "
                                            "out_features_gradient"));

        TensorShape filter_backprop_shape(filter.shape());
        Tensor* filter_backprop = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, filter_backprop_shape,
                                                &filter_backprop));

        std::vector<int> filter_dims({
                int(filter.shape().dim_size(0)),
                int(filter.shape().dim_size(1)),
                int(filter.shape().dim_size(2)),
                int(filter.shape().dim_size(3)),
                int(filter.shape().dim_size(4)),
        });

        bool individual_extents = extents.shape().dim_size(0) ==
                                          out_positions.shape().dim_size(0) &&
                                  extents.shape().dim_size(0) > 1;

        bool isotropic_extents = extents.shape().dim_size(1) == 1;

        bool point_importances = inp_importance.shape().dim_size(0) != 0;

        bool has_neighbors_importances =
                neighbors_importance.shape().dim_size(0) != 0;

        Kernel(context, filter, out_positions, extents, offset, inp_positions,
               inp_features, inp_importance, neighbors_index,
               neighbors_importance, neighbors_row_splits,
               out_features_gradient, filter_dims, individual_extents,
               isotropic_extents, point_importances, has_neighbors_importances,
               *filter_backprop);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& filter,
                        const tensorflow::Tensor& out_positions,
                        const tensorflow::Tensor& extents,
                        const tensorflow::Tensor& offset,
                        const tensorflow::Tensor& inp_positions,
                        const tensorflow::Tensor& inp_features,
                        const tensorflow::Tensor& inp_importance,
                        const tensorflow::Tensor& neighbors_index,
                        const tensorflow::Tensor& neighbors_importance,
                        const tensorflow::Tensor& neighbors_row_splits,
                        const tensorflow::Tensor& out_features_gradient,
                        const std::vector<int>& filter_dims,
                        const bool individual_extents,
                        const bool isotropic_extents,
                        const bool point_importances,
                        const bool has_neighbors_importances,
                        tensorflow::Tensor& filter_backprop) = 0;

public:
    bool align_corners;
    bool normalize;
    open3d::ml::impl::InterpolationMode interpolation;
    open3d::ml::impl::CoordinateMapping coordinate_mapping;
    int max_temp_mem_MB;
};

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

template <class TIndex>
class ContinuousConvTransposeOpKernel : public tensorflow::OpKernel {
public:
    explicit ContinuousConvTransposeOpKernel(
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

        const Tensor& out_importance = context->input(2);
        OP_REQUIRES(context,
                    out_importance.shape().dim_size(0) == 0 ||
                            out_importance.shape().dim_size(0) ==
                                    out_positions.shape().dim_size(0),
                    errors::InvalidArgument("length of out_importance must "
                                            "match the number of output points "
                                            "or must be 0"));

        const Tensor& extents = context->input(3);

        const Tensor& offset = context->input(4);
        OP_REQUIRES(context, offset.shape().dims() == 1,
                    errors::InvalidArgument("offset must be a rank 1 tensor"));
        OP_REQUIRES(context, offset.shape().dim_size(0) == 3,
                    errors::InvalidArgument("offset length must be 3"));

        const Tensor& inp_positions = context->input(5);
        OP_REQUIRES(context,
                    inp_positions.shape().dim_size(0) <=
                            std::numeric_limits<TIndex>::max(),
                    errors::InvalidArgument("Too many input points"));

        const Tensor& inp_features = context->input(6);

        // not used in forward pass
        // const Tensor& inp_neighbors_index = context->input(7);

        const Tensor& inp_neighbors_importance_sum = context->input(8);

        const Tensor& inp_neighbors_row_splits = context->input(9);

        const Tensor& neighbors_index = context->input(10);

        const Tensor& neighbors_importance = context->input(11);

        const Tensor& neighbors_row_splits = context->input(12);

        OP_REQUIRES(context, extents.shape().dims() == 2,
                    errors::InvalidArgument("extents must be a rank 2 tensor"));
        OP_REQUIRES(context,
                    extents.shape().dim_size(0) ==
                                    inp_positions.shape().dim_size(0) ||
                            extents.shape().dim_size(0) == 1,
                    errors::InvalidArgument("number of extents must match the "
                                            "number of inp_positions or must "
                                            "be 1"));
        OP_REQUIRES(context,
                    extents.shape().dim_size(1) == 3 ||
                            extents.shape().dim_size(1) == 1,
                    errors::InvalidArgument(
                            "number of components for extents must be 3 or 1"));

        OP_REQUIRES(
                context,
                inp_positions.shape().dim_size(0) ==
                        inp_features.shape().dim_size(0),
                errors::InvalidArgument("first dim of inp_positions does not "
                                        "match the first dim of inp_features"));

        OP_REQUIRES(
                context,
                inp_neighbors_importance_sum.shape().dim_size(0) ==
                                inp_positions.shape().dim_size(0) ||
                        inp_neighbors_importance_sum.shape().dim_size(0) == 0,
                errors::InvalidArgument(
                        "first dim of inp_neighbors_importance_sum does not "
                        "match the first dim of inp_positions",
                        inp_neighbors_importance_sum.shape().dim_size(0), "  ",
                        inp_positions.shape().dim_size(0)));

        OP_REQUIRES(context,
                    out_positions.shape().dim_size(0) ==
                                    out_importance.shape().dim_size(0) ||
                            out_importance.shape().dim_size(0) == 0,
                    errors::InvalidArgument("first dim of out_positions does "
                                            "not match the first dim of "
                                            "out_importance"));

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

        TensorShape out_features_shape({out_positions.shape().dim_size(0),
                                        filter.shape().dim_size(4)});
        Tensor* out_features = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_features_shape,
                                                         &out_features));

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

        bool point_importances = out_importance.shape().dim_size(0) != 0;

        bool has_neighbors_importances =
                neighbors_importance.shape().dim_size(0) != 0;

        Kernel(context, filter, out_positions, out_importance, extents, offset,
               inp_positions, inp_features, inp_neighbors_importance_sum,
               inp_neighbors_row_splits, neighbors_index, neighbors_importance,
               neighbors_row_splits, filter_dims, individual_extents,
               isotropic_extents, point_importances, has_neighbors_importances,
               *out_features);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& filter,
                        const tensorflow::Tensor& out_positions,
                        const tensorflow::Tensor& out_importance,
                        const tensorflow::Tensor& extents,
                        const tensorflow::Tensor& offset,
                        const tensorflow::Tensor& inp_positions,
                        const tensorflow::Tensor& inp_features,
                        const tensorflow::Tensor& inp_neighbors_importance_sum,
                        const tensorflow::Tensor& inp_neighbors_row_splits,
                        const tensorflow::Tensor& neighbors_index,
                        const tensorflow::Tensor& neighbors_importance,
                        const tensorflow::Tensor& neighbors_row_splits,
                        const std::vector<int>& filter_dims,
                        const bool individual_extents,
                        const bool isotropic_extents,
                        const bool point_importances,
                        const bool has_neighbors_importances,
                        tensorflow::Tensor& out_features) = 0;

public:
    bool align_corners;
    bool normalize;
    open3d::ml::impl::InterpolationMode interpolation;
    open3d::ml::impl::CoordinateMapping coordinate_mapping;
    int max_temp_mem_MB;
};

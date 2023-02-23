// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/errors.h>

#include "open3d/ml/ShapeChecking.h"

inline std::vector<open3d::ml::op_util::DimValue> GetShapeVector(
        ::tensorflow::shape_inference::InferenceContext* c,
        ::tensorflow::shape_inference::ShapeHandle shape_handle) {
    using namespace open3d::ml::op_util;
    if (!c->RankKnown(shape_handle)) {
        return std::vector<DimValue>();
    }

    std::vector<DimValue> shape;
    const int rank = c->Rank(shape_handle);
    for (int i = 0; i < rank; ++i) {
        auto d = c->DimKnownRank(shape_handle, i);
        if (c->ValueKnown(d)) {
            shape.push_back(c->Value(d));
        } else {
            shape.push_back(DimValue());
        }
    }
    return shape;
}

template <open3d::ml::op_util::CSOpt Opt = open3d::ml::op_util::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(
        ::tensorflow::shape_inference::InferenceContext* c,
        ::tensorflow::shape_inference::ShapeHandle shape_handle,
        TDimX&& dimex,
        TArgs&&... args) {
    if (!c->RankKnown(shape_handle)) {
        // without rank we cannot check
        return std::make_tuple(true, std::string());
    }
    return open3d::ml::op_util::CheckShape<Opt>(GetShapeVector(c, shape_handle),
                                                std::forward<TDimX>(dimex),
                                                std::forward<TArgs>(args)...);
}

inline std::vector<open3d::ml::op_util::DimValue> GetShapeVector(
        const tensorflow::Tensor& tensor) {
    using namespace open3d::ml::op_util;

    std::vector<DimValue> shape;
    for (int i = 0; i < tensor.dims(); ++i) {
        shape.push_back(tensor.dim_size(i));
    }
    return shape;
}

template <open3d::ml::op_util::CSOpt Opt = open3d::ml::op_util::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(const tensorflow::Tensor& tensor,
                                         TDimX&& dimex,
                                         TArgs&&... args) {
    return open3d::ml::op_util::CheckShape<Opt>(GetShapeVector(tensor),
                                                std::forward<TDimX>(dimex),
                                                std::forward<TArgs>(args)...);
}

//
// Helper function for creating a ShapeHandle from dim expressions.
// Dim expressions which are not constant will translate to unknown dims in
// the returned shape handle.
//
// Usage:
//   // ctx is of type tensorflow::shape_inference::InferenceContext*
//   {
//     using namespace open3d::ml::op_util;
//     Dim w("w");
//     Dim h("h");
//     CHECK_SHAPE_HANDLE(ctx, handle1, 10, w, h); // checks if the first dim is
//                                                 // 10 and assigns w and h
//                                                 // based on the shape of
//                                                 // handle1
//
//     CHECK_SHAPE_HANDLE(ctx, handle2, 10, 20, h); // this checks if the the
//                                           // last dim of handle2 matches the
//                                           // last dim of handle1. The first
//                                           // two dims must match 10, 20.
//
//     ShapeHandle out_shape = MakeShapeHandle(ctx, Dim(), h, w);
//     ctx->set_output(0, out_shape);
//   }
//
//
// See "../ShapeChecking.h" for more info and limitations.
//
template <class TDimX, class... TArgs>
::tensorflow::shape_inference::ShapeHandle MakeShapeHandle(
        ::tensorflow::shape_inference::InferenceContext* ctx,
        TDimX&& dimex,
        TArgs&&... args) {
    using namespace tensorflow::shape_inference;
    using namespace open3d::ml::op_util;
    std::vector<int64_t> shape = CreateDimVector(
            int64_t(InferenceContext::kUnknownDim), dimex, args...);
    std::vector<DimensionHandle> dims;
    for (int64_t d : shape) {
        dims.push_back(ctx->MakeDim(d));
    }
    return ctx->MakeShape(dims);
}

//
// Macros for checking the shape of ShapeHandle during shape inference.
//
// Usage:
//   // ctx is of type tensorflow::shape_inference::InferenceContext*
//   {
//     using namespace open3d::ml::op_util;
//     Dim w("w");
//     Dim h("h");
//     CHECK_SHAPE_HANDLE(ctx, handle1, 10, w, h); // checks if the first dim is
//                                                 // 10 and assigns w and h
//                                                 // based on the shape of
//                                                 // handle1
//
//     CHECK_SHAPE_HANDLE(ctx, handle2, 10, 20, h); // this checks if the the
//                                           // last dim of handle2 matches the
//                                           // last dim of handle1. The first
//                                           // two dims must match 10, 20.
//   }
//
//
// See "../ShapeChecking.h" for more info and limitations.
//
#define CHECK_SHAPE_HANDLE(ctx, shape_handle, ...)                           \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) =                                  \
                CheckShape(ctx, shape_handle, __VA_ARGS__);                  \
        if (TF_PREDICT_FALSE(!cs_success_)) {                                \
            return tensorflow::errors::InvalidArgument(                      \
                    "invalid shape for '" #shape_handle "', " + cs_errstr_); \
        }                                                                    \
    } while (0)

#define CHECK_SHAPE_HANDLE_COMBINE_FIRST_DIMS(ctx, shape_handle, ...)        \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) =                                  \
                CheckShape<CSOpt::COMBINE_FIRST_DIMS>(ctx, shape_handle,     \
                                                      __VA_ARGS__);          \
        if (TF_PREDICT_FALSE(!cs_success_)) {                                \
            return tensorflow::errors::InvalidArgument(                      \
                    "invalid shape for '" #shape_handle "', " + cs_errstr_); \
        }                                                                    \
    } while (0)

#define CHECK_SHAPE_HANDLE_IGNORE_FIRST_DIMS(ctx, shape_handle, ...)         \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) =                                  \
                CheckShape<CSOpt::IGNORE_FIRST_DIMS>(ctx, shape_handle,      \
                                                     __VA_ARGS__);           \
        if (TF_PREDICT_FALSE(!cs_success_)) {                                \
            return tensorflow::errors::InvalidArgument(                      \
                    "invalid shape for '" #shape_handle "', " + cs_errstr_); \
        }                                                                    \
    } while (0)

#define CHECK_SHAPE_HANDLE_COMBINE_LAST_DIMS(ctx, shape_handle, ...)         \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) =                                  \
                CheckShape<CSOpt::COMBINE_LAST_DIMS>(ctx, shape_handle,      \
                                                     __VA_ARGS__);           \
        if (TF_PREDICT_FALSE(!cs_success_)) {                                \
            return tensorflow::errors::InvalidArgument(                      \
                    "invalid shape for '" #shape_handle "', " + cs_errstr_); \
        }                                                                    \
    } while (0)

#define CHECK_SHAPE_HANDLE_IGNORE_LAST_DIMS(ctx, shape_handle, ...)          \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) =                                  \
                CheckShape<CSOpt::IGNORE_LAST_DIMS>(ctx, shape_handle,       \
                                                    __VA_ARGS__);            \
        if (TF_PREDICT_FALSE(!cs_success_)) {                                \
            return tensorflow::errors::InvalidArgument(                      \
                    "invalid shape for '" #shape_handle "', " + cs_errstr_); \
        }                                                                    \
    } while (0)

//
// Macros for checking the shape of Tensors.
// Usage:
//   // ctx is of type tensorflow::OpKernelContext*
//   {
//     using namespace open3d::ml::op_util;
//     Dim w("w");
//     Dim h("h");
//     CHECK_SHAPE(ctx, tensor1, 10, w, h); // checks if the first dim is 10
//                                          // and assigns w and h based on
//                                          // the shape of tensor1
//
//     CHECK_SHAPE(ctx, tensor2, 10, 20, h); // this checks if the the last dim
//                                           // of tensor2 matches the last dim
//                                           // of tensor1. The first two dims
//                                           // must match 10, 20.
//   }
//
//
// See "../ShapeChecking.h" for more info and limitations.
//
#define CHECK_SHAPE(ctx, tensor, ...)                                        \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) = CheckShape(tensor, __VA_ARGS__); \
        OP_REQUIRES(                                                         \
                ctx, cs_success_,                                            \
                tensorflow::errors::InvalidArgument(                         \
                        "invalid shape for '" #tensor "', " + cs_errstr_));  \
    } while (0)

#define CHECK_SHAPE_COMBINE_FIRST_DIMS(ctx, tensor, ...)                    \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::COMBINE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        OP_REQUIRES(                                                        \
                ctx, cs_success_,                                           \
                tensorflow::errors::InvalidArgument(                        \
                        "invalid shape for '" #tensor "', " + cs_errstr_)); \
    } while (0)

#define CHECK_SHAPE_IGNORE_FIRST_DIMS(ctx, tensor, ...)                     \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::IGNORE_FIRST_DIMS>(tensor, __VA_ARGS__);  \
        OP_REQUIRES(                                                        \
                ctx, cs_success_,                                           \
                tensorflow::errors::InvalidArgument(                        \
                        "invalid shape for '" #tensor "', " + cs_errstr_)); \
    } while (0)

#define CHECK_SHAPE_COMBINE_LAST_DIMS(ctx, tensor, ...)                     \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::COMBINE_LAST_DIMS>(tensor, __VA_ARGS__);  \
        OP_REQUIRES(                                                        \
                ctx, cs_success_,                                           \
                tensorflow::errors::InvalidArgument(                        \
                        "invalid shape for '" #tensor "', " + cs_errstr_)); \
    } while (0)

#define CHECK_SHAPE_IGNORE_LAST_DIMS(ctx, tensor, ...)                      \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::IGNORE_LAST_DIMS>(tensor, __VA_ARGS__);   \
        OP_REQUIRES(                                                        \
                ctx, cs_success_,                                           \
                tensorflow::errors::InvalidArgument(                        \
                        "invalid shape for '" #tensor "', " + cs_errstr_)); \
    } while (0)

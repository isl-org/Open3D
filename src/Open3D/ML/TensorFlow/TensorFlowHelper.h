// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
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
#include "../ShapeChecking.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

inline std::vector<open3d::ml::shape_checking::DimValue> GetShapeVector(
        const tensorflow::Tensor& tensor) {
    using namespace open3d::ml::shape_checking;

    std::vector<DimValue> shape;
    for (int i = 0; i < tensor.dims(); ++i) {
        shape.push_back(tensor.dim_size(i));
    }
    return shape;
}

template <open3d::ml::shape_checking::CSOpt Opt =
                  open3d::ml::shape_checking::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(const tensorflow::Tensor& tensor,
                                         TDimX&& dimex,
                                         TArgs&&... args) {
    return open3d::ml::shape_checking::CheckShape<Opt>(
            GetShapeVector(tensor), std::forward<TDimX>(dimex),
            std::forward<TArgs>(args)...);
}

// Macros for checking the shape of Tensors.
// Usage:
//   // ctx is of type tensorflow::OpKernelContext*
//   {
//     using namespace open3d::ml::shape_checking;
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

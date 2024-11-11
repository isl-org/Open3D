// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <paddle/phi/backends/stream.h>
#include <paddle/phi/common/place.h>

#include <sstream>
#include <type_traits>

#include "open3d/ml/ShapeChecking.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/allocator.h"

// Macros for checking tensor properties
#define CHECK_CUDA(x)                                      \
    do {                                                   \
        PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor"); \
    } while (0)

// NOTE: The input Tensor will be preprocessed into a contiguous Tensor within
// the execution function of the custom operator, so CHECK_CONTIGUOUS will be
// always True as there is no need for an explicit conversion in Open3D. For
// reference, please see:
// https://github.com/PaddlePaddle/Paddle/blob/65126f558a5c0fbb0cd1aa0a42844a73632ff9e9/paddle/fluid/eager/custom_operator/custom_operator_utils.cc#L803-L810
#define CHECK_CONTIGUOUS(x) \
    do {                    \
    } while (0)

#define CHECK_TYPE(x, type)                                       \
    do {                                                          \
        PD_CHECK(x.dtype() == type, #x " must have type " #type); \
    } while (0)

#define CHECK_SAME_DEVICE_TYPE(...)                                           \
    do {                                                                      \
        if (!SameDeviceType({__VA_ARGS__})) {                                 \
            PD_CHECK(false,                                                   \
                     #__VA_ARGS__                                             \
                             " must all have the same device type but got " + \
                             TensorInfoStr({__VA_ARGS__}));                   \
        }                                                                     \
    } while (0)

#define CHECK_SAME_DTYPE(...)                                                  \
    do {                                                                       \
        if (!SameDtype({__VA_ARGS__})) {                                       \
            PD_CHECK(false, #__VA_ARGS__                                       \
                                    " must all have the same dtype but got " + \
                                    TensorInfoStr({__VA_ARGS__}));             \
        }                                                                      \
    } while (0)
// Conversion from standard types to paddle types
typedef std::remove_const<decltype(paddle::DataType::INT32)>::type
        PaddleDtype_t;
template <class T>
inline PaddleDtype_t ToPaddleDtype() {
    PD_CHECK(false, "Unsupported type");
}
template <>
inline PaddleDtype_t ToPaddleDtype<uint8_t>() {
    return paddle::DataType::UINT8;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int8_t>() {
    return paddle::DataType::INT8;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int16_t>() {
    return paddle::DataType::INT16;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int32_t>() {
    return paddle::DataType::INT32;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int64_t>() {
    return paddle::DataType::INT64;
}
template <>
inline PaddleDtype_t ToPaddleDtype<float>() {
    return paddle::DataType::FLOAT32;
}
template <>
inline PaddleDtype_t ToPaddleDtype<double>() {
    return paddle::DataType::FLOAT64;
}

// convenience function for comparing standard types with paddle types
template <class T, class TDtype>
inline bool ComparePaddleDtype(const TDtype& t) {
    return ToPaddleDtype<T>() == t;
}

// convenience function to check if all tensors have the same device type
inline bool SameDeviceType(std::initializer_list<paddle::Tensor> tensors) {
    if (tensors.size()) {
        auto device_type = tensors.begin()->place().GetDeviceType();
        for (const auto& t : tensors) {
            if (device_type != t.place().GetDeviceType()) {
                return false;
            }
        }
    }
    return true;
}

// convenience function to check if all tensors have the same dtype
inline bool SameDtype(std::initializer_list<paddle::Tensor> tensors) {
    if (tensors.size()) {
        auto dtype = tensors.begin()->dtype();
        for (const auto& t : tensors) {
            if (dtype != t.dtype()) {
                return false;
            }
        }
    }
    return true;
}

inline std::string TensorInfoStr(
        std::initializer_list<paddle::Tensor> tensors) {
    std::stringstream sstr;
    size_t count = 0;
    for (const auto& t : tensors) {
        sstr << "Tensor(" << t.size() << ", " << t.place() << ")";
        ++count;
        if (count < tensors.size()) sstr << ", ";
    }
    return sstr.str();
}

// convenience function for creating a tensor for temp memory
inline paddle::Tensor CreateTempTensor(const int64_t size,
                                       const paddle::Place& device,
                                       void** ptr = nullptr) {
    paddle::Tensor tensor =
            paddle::empty({size}, ToPaddleDtype<uint8_t>(), device);
    if (ptr) {
        *ptr = tensor.data<uint8_t>();
    }
    return tensor;
}

inline std::vector<open3d::ml::op_util::DimValue> GetShapeVector(
        paddle::Tensor tensor) {
    using namespace open3d::ml::op_util;
    const auto old_shape = tensor.shape();
    std::vector<DimValue> shape;
    for (auto i = 0u; i < old_shape.size(); ++i) {
        shape.push_back(old_shape[i]);
    }
    return shape;
}

template <open3d::ml::op_util::CSOpt Opt = open3d::ml::op_util::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(paddle::Tensor tensor,
                                         TDimX&& dimex,
                                         TArgs&&... args) {
    return open3d::ml::op_util::CheckShape<Opt>(GetShapeVector(tensor),
                                                std::forward<TDimX>(dimex),
                                                std::forward<TArgs>(args)...);
}

//
// Macros for checking the shape of Tensors.
// Usage:
//   {
//     using namespace open3d::ml::op_util;
//     Dim w("w");
//     Dim h("h");
//     CHECK_SHAPE(tensor1, 10, w, h); // checks if the first dim is 10
//                                     // and assigns w and h based on
//                                     // the shape of tensor1
//
//     CHECK_SHAPE(tensor2, 10, 20, h); // this checks if the the last dim
//                                      // of tensor2 matches the last dim
//                                      // of tensor1. The first two dims
//                                      // must match 10, 20.
//   }
//
//
// See "../ShapeChecking.h" for more info and limitations.
//
#define CHECK_SHAPE(tensor, ...)                                             \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) = CheckShape(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                                \
                 "invalid shape for '" #tensor "', " + cs_errstr_);          \
    } while (0)

#define CHECK_SHAPE_COMBINE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::COMBINE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                               \
                 "invalid shape for '" #tensor "', " + cs_errstr_);         \
    } while (0)

#define CHECK_SHAPE_IGNORE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::IGNORE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                              \
                 "invalid shape for '" #tensor "', " + cs_errstr_);        \
    } while (0)

#define CHECK_SHAPE_COMBINE_LAST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::COMBINE_LAST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                              \
                 "invalid shape for '" #tensor "', " + cs_errstr_);        \
    } while (0)

#define CHECK_SHAPE_IGNORE_LAST_DIMS(tensor, ...)                         \
    do {                                                                  \
        bool cs_success_;                                                 \
        std::string cs_errstr_;                                           \
        std::tie(cs_success_, cs_errstr_) =                               \
                CheckShape<CSOpt::IGNORE_LAST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                             \
                 "invalid shape for '" #tensor "', " + cs_errstr_);       \
    } while (0)

#ifdef BUILD_CUDA_MODULE
static void cudaFreeWrapper(void* ptr) {
    phi::gpuError_t result = cudaFree(ptr);
    PADDLE_ENFORCE_GPU_SUCCESS(result);
}
#endif

// NOTE: Hack to support empty tensor, like Tensor(shape=[0], [])
template <typename T>
paddle::Tensor InitializedEmptyTensor(const phi::IntArray& shape,
                                      const phi::Place& place) {
    int64_t size = 1;
    for (auto v : shape.GetData()) {
        size *= v;
    }
    PD_CHECK(size == 0, "The numel of empty tensor is not equal to 0.");

    paddle::Deleter deleter;
    T* ptr = nullptr;
    if (phi::is_gpu_place(place)) {
#ifdef BUILD_CUDA_MODULE
        phi::gpuError_t result = cudaMalloc(&ptr, sizeof(T) * 1);
        PADDLE_ENFORCE_GPU_SUCCESS(result);
        deleter = std::function<void(void*)>(cudaFreeWrapper);
#else
        PD_CHECK(false,
                 "InitializedEmptyTensor was not compiled with CUDA support");
#endif
    } else if (phi::is_cpu_place(place)) {
        ptr = (T*)malloc(sizeof(T) * 1);
        deleter = std::function<void(void*)>(free);
    } else {
        PD_CHECK(false, "Not supported backend!");
    }

    // NOTE: In Paddle, the stride of an empty (0-size) tensor can be the same
    // as its shape.
    return paddle::from_blob(static_cast<void*>(ptr), shape, shape,
                             paddle::DataType(ToPaddleDtype<T>()),
                             phi::DataLayout::NCHW, place, deleter);
}

paddle::Tensor InitializedEmptyTensor(const phi::DataType dtype,
                                      const phi::IntArray& shape,
                                      const phi::Place& place);

// return a array of [0 1 2 ... end-1]
paddle::Tensor Arange(const int end, const paddle::Place& place);

// just like tensor.transpose(dim0,dim1)
paddle::Tensor Transpose(const paddle::Tensor& t, int64_t dim0, int64_t dim1);

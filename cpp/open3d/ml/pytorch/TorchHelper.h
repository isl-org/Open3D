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
#include <sstream>
#include <type_traits>

#include "open3d/ml/ShapeChecking.h"
#include "torch/script.h"

// Macros for checking tensor properties
#define CHECK_CUDA(x)                                         \
    do {                                                      \
        TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor") \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                      \
    do {                                                         \
        TORCH_CHECK(x.is_contiguous(), #x " must be contiguous") \
    } while (0)

#define CHECK_TYPE(x, type)                                                \
    do {                                                                   \
        TORCH_CHECK(x.dtype() == torch::type, #x " must have type " #type) \
    } while (0)

#define CHECK_SAME_DEVICE_TYPE(...)                                          \
    do {                                                                     \
        if (!SameDeviceType({__VA_ARGS__})) {                                \
            TORCH_CHECK(                                                     \
                    false,                                                   \
                    #__VA_ARGS__                                             \
                            " must all have the same device type but got " + \
                            TensorInfoStr({__VA_ARGS__}))                    \
        }                                                                    \
    } while (0)

#define CHECK_SAME_DTYPE(...)                                              \
    do {                                                                   \
        if (!SameDtype({__VA_ARGS__})) {                                   \
            TORCH_CHECK(false,                                             \
                        #__VA_ARGS__                                       \
                                " must all have the same dtype but got " + \
                                TensorInfoStr({__VA_ARGS__}))              \
        }                                                                  \
    } while (0)

// Conversion from standard types to torch types
typedef std::remove_const<decltype(torch::kInt32)>::type TorchDtype_t;
template <class T>
inline TorchDtype_t ToTorchDtype() {
    TORCH_CHECK(false, "Unsupported type");
}
template <>
inline TorchDtype_t ToTorchDtype<uint8_t>() {
    return torch::kUInt8;
}
template <>
inline TorchDtype_t ToTorchDtype<int8_t>() {
    return torch::kInt8;
}
template <>
inline TorchDtype_t ToTorchDtype<int16_t>() {
    return torch::kInt16;
}
template <>
inline TorchDtype_t ToTorchDtype<int32_t>() {
    return torch::kInt32;
}
template <>
inline TorchDtype_t ToTorchDtype<int64_t>() {
    return torch::kInt64;
}
template <>
inline TorchDtype_t ToTorchDtype<float>() {
    return torch::kFloat32;
}
template <>
inline TorchDtype_t ToTorchDtype<double>() {
    return torch::kFloat64;
}

// convenience function for comparing standard types with torch types
template <class T, class TDtype>
inline bool CompareTorchDtype(const TDtype& t) {
    return ToTorchDtype<T>() == t;
}

// convenience function to check if all tensors have the same device type
inline bool SameDeviceType(std::initializer_list<torch::Tensor> tensors) {
    if (tensors.size()) {
        auto device_type = tensors.begin()->device().type();
        for (auto t : tensors) {
            if (device_type != t.device().type()) {
                return false;
            }
        }
    }
    return true;
}

// convenience function to check if all tensors have the same dtype
inline bool SameDtype(std::initializer_list<torch::Tensor> tensors) {
    if (tensors.size()) {
        auto device_type = tensors.begin()->dtype();
        for (auto t : tensors) {
            if (device_type != t.dtype()) {
                return false;
            }
        }
    }
    return true;
}

inline std::string TensorInfoStr(std::initializer_list<torch::Tensor> tensors) {
    std::stringstream sstr;
    size_t count = 0;
    for (const auto t : tensors) {
        sstr << t.sizes() << " " << t.toString() << " " << t.device();
        ++count;
        if (count < tensors.size()) sstr << ", ";
    }
    return sstr.str();
}

// convenience function for creating a tensor for temp memory
inline torch::Tensor CreateTempTensor(const int64_t size,
                                      const torch::Device& device,
                                      void** ptr = nullptr) {
    torch::Tensor tensor = torch::empty(
            {size}, torch::dtype(ToTorchDtype<uint8_t>()).device(device));
    if (ptr) {
        *ptr = tensor.data_ptr<uint8_t>();
    }
    return tensor;
}

inline std::vector<open3d::ml::op_util::DimValue> GetShapeVector(
        torch::Tensor tensor) {
    using namespace open3d::ml::op_util;

    std::vector<DimValue> shape;
    const int rank = tensor.dim();
    for (int i = 0; i < rank; ++i) {
        shape.push_back(tensor.size(i));
    }
    return shape;
}

template <open3d::ml::op_util::CSOpt Opt = open3d::ml::op_util::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(torch::Tensor tensor,
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
        TORCH_CHECK(cs_success_,                                             \
                    "invalid shape for '" #tensor "', " + cs_errstr_)        \
    } while (0)

#define CHECK_SHAPE_COMBINE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::COMBINE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        TORCH_CHECK(cs_success_,                                            \
                    "invalid shape for '" #tensor "', " + cs_errstr_)       \
    } while (0)

#define CHECK_SHAPE_IGNORE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::IGNORE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        TORCH_CHECK(cs_success_,                                           \
                    "invalid shape for '" #tensor "', " + cs_errstr_)      \
    } while (0)

#define CHECK_SHAPE_COMBINE_LAST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::COMBINE_LAST_DIMS>(tensor, __VA_ARGS__); \
        TORCH_CHECK(cs_success_,                                           \
                    "invalid shape for '" #tensor "', " + cs_errstr_)      \
    } while (0)

#define CHECK_SHAPE_IGNORE_LAST_DIMS(tensor, ...)                         \
    do {                                                                  \
        bool cs_success_;                                                 \
        std::string cs_errstr_;                                           \
        std::tie(cs_success_, cs_errstr_) =                               \
                CheckShape<CSOpt::IGNORE_LAST_DIMS>(tensor, __VA_ARGS__); \
        TORCH_CHECK(cs_success_,                                          \
                    "invalid shape for '" #tensor "', " + cs_errstr_)     \
    } while (0)

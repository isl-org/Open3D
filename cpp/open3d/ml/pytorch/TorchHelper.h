// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// https://stackoverflow.com/q/77034039 : False Alarm warnings from PyTorch
// headers
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <torch/script.h>

#include <sstream>
#include <type_traits>

#include "open3d/core/Device.h"
#include "open3d/ml/ShapeChecking.h"
#include "open3d/utility/Logging.h"

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
        for (const auto& t : tensors) {
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
        auto dtype = tensors.begin()->dtype();
        for (const auto& t : tensors) {
            if (dtype != t.dtype()) {
                return false;
            }
        }
    }
    return true;
}

inline std::string TensorInfoStr(std::initializer_list<torch::Tensor> tensors) {
    std::stringstream sstr;
    size_t count = 0;
    for (const auto& t : tensors) {
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

// allow_tf32 (Intel XMX/NVIDIA tensor-core reduced-precision GEMM) is only
// implemented for the SYCL conv-op backend; CPU and CUDA always compute in
// full precision regardless of this flag. Call once from each CPU/CUDA conv
// op's entry point to warn the user their request is silently ignored there.
inline void WarnIfTF32NotSupported(bool allow_tf32) {
    if (allow_tf32) {
        open3d::utility::LogWarning(
                "allow_tf32 is not supported on this backend; computing in "
                "full float32 precision instead.");
    }
}

// Runs the shared SYCL conv-op two-pass temp-memory pattern: call `run_fn`
// once with temp==nullptr to query the required size, allocate a temp
// tensor sized by max_temp_mem_MB, then call `run_fn` again to actually run
// the op. `run_fn` is `(void* temp, size_t& temp_size, size_t&
// max_temp_size) -> void` and is expected to forward these straight to the
// underlying `*ComputeFeaturesSYCL` function. Factoring out this pattern
// avoids duplicating each op's full (15-20 argument) call site twice.
template <class Fn>
inline void RunSYCLWithTempMemory(const torch::Device& device,
                                  int64_t max_temp_mem_MB,
                                  Fn&& run_fn) {
    void* temp_ptr = nullptr;
    size_t temp_size = 0;
    size_t max_temp_size = 0;

    // determine temp_size
    run_fn(temp_ptr, temp_size, max_temp_size);

    temp_size = std::max(
            std::min(size_t(max_temp_mem_MB) * 1024 * 1024, max_temp_size),
            temp_size);

    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually run the operation
    run_fn(temp_ptr, temp_size, max_temp_size);
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

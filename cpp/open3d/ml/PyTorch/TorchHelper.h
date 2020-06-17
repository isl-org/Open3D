#pragma once

#include <torch/script.h>
#include <type_traits>

// Macros for checking tensor properties
#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_TYPE(x, type) \
    TORCH_CHECK(x.dtype() == torch::type, #x " must have type " #type)

#define CHECK_SAME_DEVICE_TYPE(...)            \
    TORCH_CHECK(SameDeviceType({__VA_ARGS__}), \
                #__VA_ARGS__ " must all have the same device type")

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

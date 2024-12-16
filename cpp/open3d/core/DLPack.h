// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// This file is retrieved from:
// https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
// Commit: 3ec0443, Feb 16 2020
//
// License:
// https://github.com/dmlc/dlpack/blob/master/LICENSE
//
// Open3D changes:
// No changes except for automatic style changed by clang-format.

/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current version of dlpack */
#define DLPACK_VERSION 020

/*! \brief DLPACK_DLL prefix for windows */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/*!
 * \brief The device type in DLContext.
 */
typedef enum {
    /*! \brief CPU device */
    kDLCPU = 1,
    /*! \brief CUDA GPU device */
    kDLGPU = 2,
    /*!
     * \brief Pinned CUDA GPU device by cudaMallocHost
     * \note kDLCPUPinned = kDLCPU | kDLGPU
     */
    kDLCPUPinned = 3,
    /*! \brief OpenCL devices. */
    kDLOpenCL = 4,
    /*! \brief Vulkan buffer for next generation graphics. */
    kDLVulkan = 7,
    /*! \brief Metal for Apple GPU. */
    kDLMetal = 8,
    /*! \brief Verilog simulator buffer */
    kDLVPI = 9,
    /*! \brief ROCm GPUs for AMD GPUs */
    kDLROCM = 10,
    /*!
     * \brief Reserved extension device type,
     * used for quickly test extension device
     * The semantics can differ depending on the implementation.
     */
    kDLExtDev = 12,
} DLDeviceType;

/*!
 * \brief A Device context for Tensor and operator.
 */
typedef struct {
    /*! \brief The device type used in the device. */
    DLDeviceType device_type;
    /*! \brief The device index */
    int device_id;
} DLContext;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    kDLBfloat = 4U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct {
    /*!
     * \brief Type code of base types.
     * We keep it uint8_t instead of DLDataTypeCode for minimal memory
     * footprint, but the value should be one of DLDataTypeCode enum values.
     * */
    uint8_t code;
    /*!
     * \brief Number of bits, common choices are 8, 16, 32.
     */
    uint8_t bits;
    /*! \brief Number of lanes in the type, used for vector types. */
    uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
    /*!
     * \brief The opaque data pointer points to the allocated data. This will be
     * CUDA device pointer or cl_mem handle in OpenCL. This pointer is always
     * aligned to 256 bytes as in CUDA.
     *
     * For given DLTensor, the size of memory required to store the contents of
     * data is calculated as follows:
     *
     * \code{.c}
     * static inline size_t GetDataSize(const DLTensor* t) {
     *   size_t size = 1;
     *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
     *     size *= t->shape[i];
     *   }
     *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
     *   return size;
     * }
     * \endcode
     */
    void* data;
    /*! \brief The device context of the tensor */
    DLContext ctx;
    /*! \brief Number of dimensions */
    int ndim;
    /*! \brief The data type of the pointer*/
    DLDataType dtype;
    /*! \brief The shape of the tensor */
    int64_t* shape;
    /*!
     * \brief strides of the tensor (in number of elements, not bytes)
     *  can be NULL, indicating tensor is compact and row-majored.
     */
    int64_t* strides;
    /*! \brief The offset in bytes to the beginning pointer to data */
    uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, manage memory of DLTensor. This data structure is
 *  intended to facilitate the borrowing of DLTensor by another framework. It is
 *  not meant to transfer the tensor. When the borrowing framework doesn't need
 *  the tensor, it should call the deleter to notify the host that the resource
 *  is no longer needed.
 */
typedef struct DLManagedTensor {
    /*! \brief DLTensor which is being memory managed */
    DLTensor dl_tensor;
    /*! \brief the context of the original host framework of DLManagedTensor in
     *   which DLManagedTensor is used in the framework. It can also be NULL.
     */
    void* manager_ctx;
    /*! \brief Destructor signature void (*)(void*) - this should be called
     *   to destruct manager_ctx which holds the DLManagedTensor. It can be NULL
     *   if there is no way for the caller to provide a reasonable destructor.
     *   The destructors deletes the argument self as well.
     */
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;
#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif

#include <fmt/core.h>
#include <fmt/format.h>

namespace fmt {

template <>
struct formatter<DLDeviceType> {
    template <typename FormatContext>
    auto format(const DLDeviceType& c, FormatContext& ctx) const
            -> decltype(ctx.out()) {
        const char* text = nullptr;
        switch (c) {
            case kDLCPU:
                text = "kDLCPU";
                break;
            case kDLGPU:
                text = "kDLGPU";
                break;
            case kDLCPUPinned:
                text = "kDLCPUPinned";
                break;
            case kDLOpenCL:
                text = "kDLOpenCL";
                break;
            case kDLVulkan:
                text = "kDLVulkan";
                break;
            case kDLMetal:
                text = "kDLMetal";
                break;
            case kDLVPI:
                text = "kDLVPI";
                break;
            case kDLROCM:
                text = "kDLROCM";
                break;
            case kDLExtDev:
                text = "kDLExtDev";
                break;
        }
        return format_to(ctx.out(), text);
    }

    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};

}  // namespace fmt

#endif  // DLPACK_DLPACK_H_

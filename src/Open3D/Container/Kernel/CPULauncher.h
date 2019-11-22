// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include <cassert>
#include <vector>
#include "Open3D/Container/Kernel/Scheduler.h"
#include "Open3D/Container/Tensor.h"
namespace open3d {
namespace kernel {

class CPULauncher {
public:
    template <typename scalar_t, typename func_t>
    static void LaunchUnaryEWKernel(const Tensor& src,
                                    Tensor& dst,
                                    func_t element_kernel) {
        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        int64_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());

        // src - (broadcast) -> mid -> dst
        SizeVector mid_shape = dst.GetShape();
        SizeVector mid_stride = Tensor::DefaultStrides(dst.GetShape());
        OffsetBroadcastCalculator src_offset_calculator(
                src.GetShape(), src.GetStrides(), mid_shape, mid_stride);
        OffsetBroadcastCalculator dst_offset_calculator(
                dst.GetShape(), dst.GetStrides(), mid_shape, mid_stride);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t thread_idx = 0;
             thread_idx < static_cast<int64_t>(dst.NumElements());
             thread_idx++) {
            int64_t src_idx = src_offset_calculator.GetOffset(thread_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            int64_t dst_idx = dst_offset_calculator.GetOffset(thread_idx);
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        }
    }

    // dst = src[index_tensors]
    template <typename scalar_t, typename func_t>
    static void LaunchRhsIndexedUnaryEWKernel(
            const Tensor& src,
            Tensor& dst,
            const std::vector<Tensor>& index_tensors,
            const SizeVector& indexed_out_shape,
            func_t element_kernel) {
        std::vector<const int64_t*> index_tensor_data_ptrs;
        for (auto& index : index_tensors) {
            auto index_tensor_ptr =
                    static_cast<const int64_t*>(index.GetDataPtr());
            index_tensor_data_ptrs.push_back(index_tensor_ptr);
        }

        std::vector<bool> is_trivial_dims;
        for (const Tensor& index_tensor : index_tensors) {
            is_trivial_dims.push_back(index_tensor.NumDims() == 0);
        }

        auto default_strides = Tensor::DefaultStrides(dst.GetShape());

        // [src] --fancy idx--> [mid] --broadcast--> [dst]
        SizeVector mid_shape = indexed_out_shape;
        SizeVector mid_strides = Tensor::DefaultStrides(mid_shape);
        IndexedOffsetCalculator fancy_offset_calculator(
                src.GetShape(), src.GetStrides(), mid_strides, is_trivial_dims,
                index_tensor_data_ptrs);
        OffsetBroadcastCalculator broadcast_offset_calculator(
                mid_shape, mid_strides, dst.GetShape(),
                Tensor::DefaultStrides(dst.GetShape()));
        OffsetBroadcastCalculator dst_offset_calculator(
                dst.GetShape(), dst.GetStrides(), dst.GetShape(),
                Tensor::DefaultStrides(dst.GetShape()));

        int64_t num_elems = static_cast<int64_t>(dst.GetShape().NumElements());
        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        int64_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t thread_idx = 0; thread_idx < num_elems; thread_idx++) {
            // [thread_idx] --un-broadcast--> [mid_idx] --un-fancy--> [src_idx]
            int64_t mid_idx = broadcast_offset_calculator.GetOffset(thread_idx);
            int64_t src_idx = fancy_offset_calculator.GetOffset(mid_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            int64_t dst_idx = dst_offset_calculator.GetOffset(thread_idx);
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        }
    }

    // dst[index_tensors] = src
    template <typename scalar_t, typename func_t>
    static void LaunchLhsIndexedUnaryEWKernel(
            const Tensor& src,
            Tensor& dst,
            const std::vector<Tensor>& index_tensors,
            const SizeVector& indexed_out_shape,
            func_t element_kernel) {
        std::vector<const int64_t*> index_tensor_data_ptrs;
        for (auto& index : index_tensors) {
            auto index_tensor_ptr =
                    static_cast<const int64_t*>(index.GetDataPtr());
            index_tensor_data_ptrs.push_back(index_tensor_ptr);
        }

        std::vector<bool> is_trivial_dims;
        for (const Tensor& index_tensor : index_tensors) {
            is_trivial_dims.push_back(index_tensor.NumDims() == 0);
        }

        // [src] --broadcast--> [mid] --fancy idx--> [dst]
        SizeVector mid_shape = indexed_out_shape;
        SizeVector mid_strides = Tensor::DefaultStrides(mid_shape);
        OffsetBroadcastCalculator src_offset_calculator(
                src.GetShape(), src.GetStrides(), mid_shape, mid_strides);
        IndexedOffsetCalculator dst_offset_calculator(
                dst.GetShape(), dst.GetStrides(), mid_strides, is_trivial_dims,
                index_tensor_data_ptrs);

        int64_t num_elems = static_cast<int64_t>(mid_shape.NumElements());
        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        int64_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t thread_idx = 0; thread_idx < num_elems; thread_idx++) {
            int64_t src_idx = src_offset_calculator.GetOffset(thread_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            int64_t dst_idx = dst_offset_calculator.GetOffset(thread_idx);
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        }
    }
};

}  // namespace kernel
}  // namespace open3d

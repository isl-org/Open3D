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
        size_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());
        SizeVector default_strides = Tensor::DefaultStrides(src.GetShape());
        OffsetCalculator src_offset_calculator(src.GetStrides(),
                                               default_strides);
        OffsetCalculator dst_offset_calculator(dst.GetStrides(),
                                               default_strides);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t dst_raw_idx = 0;
             dst_raw_idx < static_cast<int64_t>(src.GetShape().NumElements());
             dst_raw_idx++) {
            size_t src_idx = src_offset_calculator.GetOffset(dst_raw_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            size_t dst_idx = dst_offset_calculator.GetOffset(dst_raw_idx);
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        }
    }

    template <typename scalar_t, typename func_t>
    static void LaunchIndexedUnaryEWKernel(const Tensor& src,
                                           Tensor& dst,
                                           const std::vector<Tensor>& indices,
                                           const SizeVector& indexing_shapes,
                                           func_t element_kernel) {
        utility::LogInfo("IndexedKernel!");

        std::vector<const int*> indexing_tensor_data_ptrs;
        utility::LogInfo("src.GetStrides() = {}", src.GetStrides());
        utility::LogInfo("dst.GetStrides() = {}", dst.GetStrides());
        for (int i = 0; i < indices.size(); ++i) {
            utility::LogInfo("indices[{}] {}", i, indices[i].ToString());
            auto index_tensor_ptr =
                    static_cast<const int*>(indices[i].GetDataPtr());

            std::vector<int> tmp;
            for (auto j = 0; j < indexing_shapes[i]; ++j) {
                tmp.push_back(index_tensor_ptr[j]);
            }
            utility::LogInfo("index_tensor_ptr: {}", tmp);

            indexing_tensor_data_ptrs.push_back(index_tensor_ptr);
        }

        IndexedOffsetCalculator src_offset_calculator(
                src.GetStrides(), src.GetShape(), dst.GetStrides(),
                indexing_shapes, indexing_tensor_data_ptrs);

        int64_t num_elems = static_cast<int64_t>(dst.GetShape().NumElements());
        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        size_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t dst_idx = 0; dst_idx < num_elems; dst_idx++) {
            size_t src_idx = src_offset_calculator.GetOffset(dst_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        }
    }
};

}  // namespace kernel
}  // namespace open3d

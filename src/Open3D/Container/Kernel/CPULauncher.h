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
#include <vector>

#include "Open3D/Container/Tensor.h"

namespace open3d {
namespace kernel {

class CPULauncher {
public:
    /// Recover src tensor element offsets given dst tensor element offsets
    /// src and dst tensors have the same size but different strides
    class OffsetCalculator {
    public:
        OffsetCalculator(const std::vector<size_t>& src_strides,
                         const std::vector<size_t>& dst_strides)
            : num_dims_(src_strides.size()),
              src_strides_(src_strides),
              dst_strides_(dst_strides) {}

        size_t GetOffset(size_t dst_idx) const {
            size_t src_idx = 0;
            for (size_t dim = 0; dim < num_dims_; dim++) {
                src_idx += dst_idx / dst_strides_[dim] * src_strides_[dim];
                dst_idx = dst_idx % dst_strides_[dim];
            }
            return src_idx;
        }

    protected:
        size_t num_dims_;
        std::vector<size_t> src_strides_;
        std::vector<size_t> dst_strides_;
    };

public:
    template <typename scalar_t, typename func_t>
    static void LaunchUnaryEWKernel(const Tensor& src,
                                    Tensor& dst,
                                    const func_t& element_kernel) {
        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        size_t element_byte_size = DtypeUtil::ByteSize(src.GetDtype());
        OffsetCalculator offset_calculator(src.GetStrides(), dst.GetStrides());

        auto f = [=](size_t dst_idx) {
            size_t src_idx = offset_calculator.GetOffset(dst_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t dst_idx = 0;
             dst_idx < static_cast<int64_t>(src.GetShape().NumElements());
             dst_idx++) {
            f(dst_idx);
        }
    }
};

}  // namespace kernel
}  // namespace open3d

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
#include <tbb/parallel_for.h>

#include "open3d/core/SizeVector.h"
#include "open3d/core/kernel/CumSum.h"
#include "open3d/utility/Console.h"
#include "tbb/parallel_scan.h"

namespace open3d {
namespace core {
namespace kernel {

namespace {
class ScanSumBody {
    Tensor* out;
    const Tensor* in;
    int64_t dim;
    Tensor sum;

public:
    ScanSumBody(Tensor* out_, const Tensor* in_, int64_t dim_)
        : out(out_), in(in_), dim(dim_) {
        sum = Tensor::Zeros(
                shape_util::ReductionShape(in->GetShape(), {dim}, true),
                in->GetDtype(), in->GetDevice());
    }
    template <class Tag>
    void operator()(const tbb::blocked_range<size_t>& r, Tag) {
        Tensor temp = sum;
        for (size_t i = r.begin(); i < r.end(); ++i) {
            temp.Add_(in->Slice(dim, i, i + 1));
            if (Tag::is_final_scan()) {
                out->Slice(dim, i, i + 1).AsRvalue() = temp;
            }
        }
        sum = temp;
    }
    ScanSumBody(ScanSumBody& b, tbb::split) : out(b.out), in(b.in), dim(b.dim) {
        sum = Tensor::Zeros(
                shape_util::ReductionShape(in->GetShape(), {dim}, true),
                in->GetDtype(), in->GetDevice());
    }
    void reverse_join(ScanSumBody& a) { sum = a.sum.Add(sum); }
    void assign(ScanSumBody& b) { sum = b.sum; }
};
}  // namespace

void CumSumCPU(const Tensor& src, Tensor& dst, int64_t dim) {
    // Copy first slice of source Tensor to destination Tensor.
    dst.Slice(dim, 0, 1).CopyFrom(src.Slice(dim, 0, 1));
    int64_t num_elements = src.GetShapeRef()[dim];

    // Return if there are no elements;
    if (num_elements <= 0) {
        return;
    }

    // Parallel scan.
    ScanSumBody body(&dst, &src, dim);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, num_elements), body);
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d

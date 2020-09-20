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
#include <cstddef>

namespace tbb {
// class split;

template <typename Value>
class blocked_range;

}  // namespace tbb

namespace open3d {
namespace utility {

namespace {
template <class Tin, class Tout>
class ScanSumBody {
    Tout sum;
    const Tin* in;
    Tout* const out;

public:
    ScanSumBody(Tout* out_, const Tin* in_) : sum(0), in(in_), out(out_) {}
    // ScanSumBody(ScanSumBody& b, tbb::split): sum(0), in(b.in), out(b.out) {}
    Tout get_sum() const;

    template <class Tag>
    void operator()(const tbb::blocked_range<size_t>& r, Tag);
    void reverse_join(ScanSumBody& a);
    void assign(ScanSumBody& b);
};
}  // namespace

template <class Tin, class Tout>
void InclusivePrefixSum(const Tin* first, const Tin* last, Tout* out);

}  // namespace utility
}  // namespace open3d

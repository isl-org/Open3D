// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

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
    Tout get_sum() const { return sum; }

    template <class Tag>
    void operator()(const tbb::blocked_range<size_t>& r, Tag) {
        Tout temp = sum;
        for (size_t i = r.begin(); i < r.end(); ++i) {
            temp = temp + in[i];
            if (Tag::is_final_scan()) out[i] = temp;
        }
        sum = temp;
    }
    ScanSumBody(ScanSumBody& b, tbb::split) : sum(0), in(b.in), out(b.out) {}
    void reverse_join(ScanSumBody& a) { sum = a.sum + sum; }
    void assign(ScanSumBody& b) { sum = b.sum; }
};
}  // namespace

template <class Tin, class Tout>
void InclusivePrefixSum(const Tin* first, const Tin* last, Tout* out) {
    ScanSumBody<Tin, Tout> body(out, first);
    size_t n = std::distance(first, last);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), body);
}

}  // namespace utility
}  // namespace open3d

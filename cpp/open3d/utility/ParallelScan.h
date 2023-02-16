// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

// clang-format off
#if TBB_INTERFACE_VERSION >= 10000
    #ifdef OPEN3D_USE_ONEAPI_PACKAGES
        #include <oneapi/dpl/execution>
        #include <oneapi/dpl/numeric>
    #else
        // Check if the C++ standard library implements parallel algorithms
        // and use this over parallelstl to avoid conflicts.
        // Clang does not implement it so far, so checking for C++17 is not sufficient.
        #ifdef __cpp_lib_parallel_algorithm
            #include <execution>
            #include <numeric>
        #else
            #include <pstl/execution>
            #include <pstl/numeric>
            // parallelstl incorrectly assumes MSVC to unconditionally implement
            // parallel algorithms even if __cpp_lib_parallel_algorithm is not
            // defined. So manually include the header which pulls all
            // "pstl::execution" definitions into the "std" namespace.
            #if __PSTL_CPP17_EXECUTION_POLICIES_PRESENT
                #include <pstl/internal/glue_execution_defs.h>
            #endif
        #endif
    #endif
#endif

// clang-format on

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
#if TBB_INTERFACE_VERSION >= 10000
    // use parallelstl if we have TBB 2018 or later
#ifdef OPEN3D_USE_ONEAPI_PACKAGES
    std::inclusive_scan(oneapi::dpl::execution::par_unseq, first, last, out);

#else
    std::inclusive_scan(std::execution::par_unseq, first, last, out);
#endif
#else
    ScanSumBody<Tin, Tout> body(out, first);
    size_t n = std::distance(first, last);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), body);
#endif
}

}  // namespace utility
}  // namespace open3d

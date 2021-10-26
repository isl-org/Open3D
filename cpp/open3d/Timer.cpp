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

#include "open3d/Timer.h"

#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

namespace open3d {
namespace utility {

void RunSYCLDemo() {
    using namespace sycl;

    std::cout << "Hello RunSYCLDemo()" << std::endl;

    constexpr int size = 16;
    std::array<int, size> data;
    // Create queue on implementation-chosen default device
    sycl::queue Q;
    // Create buffer using host allocated "data" array
    buffer B{data};
    Q.submit([&](handler &h) {
        accessor A{B, h};
        h.parallel_for(size, [=](auto &idx) { A[idx] = idx; });
    });
    // Obtain access to buffer on the host
    // Will wait for device kernel to execute to generate data
    host_accessor A{B};
    for (int i = 0; i < size; i++) {
        std::cout << "data[" << i << "] = " << A[i] << "\n";
    }
}

}  // namespace utility
}  // namespace open3d

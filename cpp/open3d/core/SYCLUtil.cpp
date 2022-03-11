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

#include "open3d/core/SYCLUtil.h"

#include <CL/sycl.hpp>
#include <array>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void RunSYCLDemo() {
    utility::LogInfo("Hello RunSYCLDemo()!");

    // Creating buffer of 4 ints to be used inside the kernel code
    cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);

    // Creating SYCL queue
    cl::sycl::queue Queue;

    // Size of index space for kernel
    cl::sycl::range<1> NumOfWorkItems{Buffer.size()};

    // Submitting command group(work) to queue
    Queue.submit([&](cl::sycl::handler& cgh) {
        // Getting write only access to the buffer on a device
        auto Accessor = Buffer.get_access<cl::sycl::access::mode::write>(cgh);
        // Executing kernel
        cgh.parallel_for<class FillBuffer>(
                NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
                    // Fill buffer with indexes
                    Accessor[WIid] = (cl::sycl::cl_int)WIid.get(0);
                });
    });

    // Getting read only access to the buffer on the host.
    // Implicit barrier waiting for queue to complete the work.
    const auto HostAccessor = Buffer.get_access<cl::sycl::access::mode::read>();

    // Check the results
    bool MismatchFound = false;
    for (size_t I = 0; I < Buffer.size(); ++I) {
        if (HostAccessor[I] != I) {
            std::cout << "The result is incorrect for element: " << I
                      << " , expected: " << I << " , got: " << HostAccessor[I]
                      << std::endl;
            MismatchFound = true;
        }
    }

    if (!MismatchFound) {
        std::cout << "The results are correct!" << std::endl;
    } else {
        std::cout << "The results are not correct!" << std::endl;
    }
}

}  // namespace core
}  // namespace open3d

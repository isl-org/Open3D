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

#include "open3d/core/TensorFunction.h"

#include <CL/sycl.hpp>
#include <array>

namespace open3d {
namespace core {

void RunSYCLDemo() {
    using namespace sycl;

    utility::LogInfo("Hello world!");

    constexpr int size = 16;
    std::array<int, size> data;
    // Create queue on implementation-chosen default device
    sycl::queue Q;
    // Create buffer using host allocated "data" array
    buffer B{data};
    Q.submit([&](handler& h) {
        accessor A{B, h};
        h.parallel_for(size, [=](auto& idx) { A[idx] = idx; });
    });
    // Obtain access to buffer on the host
    // Will wait for device kernel to execute to generate data
    host_accessor A{B};
    for (int i = 0; i < size; i++) {
        std::cout << "data[" << i << "] = " << A[i] << "\n";
    }

    // // Creating buffer of 4 ints to be used inside the kernel code
    // cl::sycl::buffer<cl::sycl::cl_int, 1> buffer(1024 * 1024 * 1024);

    // // Creating SYCL queue
    // cl::sycl::queue queue;

    // // Size of index space for kernel
    // cl::sycl::range<1> num_work_items{buffer.size()};

    // // Submitting command group(work) to queue
    // queue.submit([&](cl::sycl::handler& cgh) {
    //     // Getting write only access to the buffer on a device
    //     auto accessor =
    //     buffer.get_access<cl::sycl::access::mode::write>(cgh);
    //     // Executing kernel
    //     cgh.parallel_for<class FillBuffer>(
    //             num_work_items, [=](cl::sycl::id<1> wlid) {
    //                 // Fill buffer with indexes
    //                 accessor[wlid] = (cl::sycl::cl_int)wlid.get(0);
    //             });
    // });

    // // Getting read only access to the buffer on the host.
    // // Implicit barrier waiting for queue to complete the work.
    // const auto host_accessor =
    //         buffer.get_access<cl::sycl::access::mode::read>();

    // // Check the results
    // bool mispatch_found = false;
    // for (size_t i = 0; i < buffer.size(); ++i) {
    //     if (host_accessor[i] != i) {
    //         std::cout << "The result is incorrect for element: " << i
    //                   << " , expected: " << i << " , got: " <<
    //                   host_accessor[i]
    //                   << std::endl;
    //         mispatch_found = true;
    //     }
    // }

    // if (!mispatch_found) {
    //     std::cout << "The results are correct!" << std::endl;
    // }
}

static Tensor StackAlongAxis(const Tensor& self,
                             const Tensor& other,
                             int64_t axis) {
    SizeVector combined_shape = self.GetShape();
    combined_shape[axis] += other.GetShape(axis);

    std::vector<TensorKey> tks_self;
    std::vector<TensorKey> tks_other;
    for (int i = 0; i < axis; ++i) {
        tks_self.push_back(core::TensorKey::Slice(0, self.GetShape(i), 1));
        tks_other.push_back(core::TensorKey::Slice(0, self.GetShape(i), 1));
    }

    tks_self.push_back(core::TensorKey::Slice(0, self.GetShape(axis), 1));
    tks_other.push_back(core::TensorKey::Slice(self.GetShape(axis),
                                               combined_shape[axis], 1));

    Tensor combined_tensor(combined_shape, self.GetDtype(), self.GetDevice());
    combined_tensor.SetItem(tks_self, self);
    combined_tensor.SetItem(tks_other, other);

    return combined_tensor;
}

Tensor Append(const Tensor& self,
              const Tensor& other,
              const utility::optional<int64_t> axis) {
    core::AssertTensorDevice(other, self.GetDevice());
    core::AssertTensorDtype(other, self.GetDtype());

    if (self.NumDims() != other.NumDims()) {
        utility::LogError(
                "All the input tensors must have same number of "
                "dimensions, but the tensor at index 0 has {} dimension(s) "
                "and the tensor at index 1 has {} dimension(s).",
                self.NumDims(), other.NumDims());
    }

    if (!axis.has_value()) {
        return StackAlongAxis(self.Reshape({self.NumElements(), 1}),
                              other.Reshape({other.NumElements(), 1}), 0)
                .Reshape({-1});
    } else {
        if (self.NumDims() == 0) {
            utility::LogError(
                    "Zero-dimensional tensor can only be appended along axis = "
                    "null, but got {}.",
                    axis.value());
        }

        const int64_t axis_d =
                shape_util::WrapDim(axis.value(), self.NumDims());

        for (int64_t i = 0; i < self.NumDims(); ++i) {
            if (i != axis_d && self.GetShape(i) != other.GetShape(i)) {
                utility::LogError(
                        "All the input tensor dimensions, other than dimension "
                        "size along concatenation axis must be same, but along "
                        "dimension {}, the tensor at index 0 has size {} and "
                        "the tensor at index 1 has size {}.",
                        i, self.GetShape(i), other.GetShape(i));
            }
        }

        return StackAlongAxis(self, other, axis_d);
    }
}

}  // namespace core
}  // namespace open3d

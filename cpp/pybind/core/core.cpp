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

#include "pybind/core/core.h"

#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

Tensor PyArrayToTensor(py::array array, bool inplace) {
    py::buffer_info info = array.request();

    SizeVector shape(info.shape.begin(), info.shape.end());
    SizeVector strides(info.strides.begin(), info.strides.end());
    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] /= info.itemsize;
    }
    Dtype dtype = pybind_utils::ArrayFormatToDtype(info.format);
    Device device("CPU:0");

    std::function<void(void *)> deleter = [](void *) -> void {};
    auto blob = std::make_shared<Blob>(device, info.ptr, deleter);
    Tensor t_inplace(shape, strides, info.ptr, dtype, blob);

    if (inplace) {
        return t_inplace;
    } else {
        return t_inplace.Copy();
    }
}

void pybind_core(py::module &m) {
    py::module m_core = m.def_submodule("core");
    pybind_cuda_utils(m_core);
    pybind_core_blob(m_core);
    pybind_core_dtype(m_core);
    pybind_core_device(m_core);
    pybind_core_size_vector(m_core);
    pybind_core_tensor_key(m_core);
    pybind_core_tensor(m_core);
    pybind_core_tensorlist(m_core);
    pybind_core_linalg(m_core);
    pybind_core_kernel(m_core);
    pybind_core_nn(m_core);
}

}  // namespace core
}  // namespace open3d

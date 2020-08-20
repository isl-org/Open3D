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

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {

void pybind_core_tensor_key(py::module& m) {
    py::class_<core::NoneType> none_type(m, "NoneType", "Open3D None type.");
    none_type.def(py::init([]() { return new core::NoneType(); }));

    py::class_<core::TensorKey> tensor_key(m, "TensorKey");
    tensor_key.def("get_start", &core::TensorKey::GetStart)
            .def("get_stop", &core::TensorKey::GetStop)
            .def("get_step", &core::TensorKey::GetStep);

    tensor_key.def_static("index", &core::TensorKey::Index);
    tensor_key.def_static("slice",
                          [](int64_t start, int64_t stop, int64_t step) {
                              return core::TensorKey::Slice(start, stop, step);
                          });
    tensor_key.def_static("slice",
                          [](int64_t start, int64_t stop, core::NoneType step) {
                              return core::TensorKey::Slice(start, stop, step);
                          });
    tensor_key.def_static("slice",
                          [](int64_t start, core::NoneType stop, int64_t step) {
                              return core::TensorKey::Slice(start, stop, step);
                          });
    tensor_key.def_static("slice", [](int64_t start, core::NoneType stop,
                                      core::NoneType step) {
        return core::TensorKey::Slice(start, stop, step);
    });
    tensor_key.def_static("slice",
                          [](core::NoneType start, int64_t stop, int64_t step) {
                              return core::TensorKey::Slice(start, stop, step);
                          });
    tensor_key.def_static("slice", [](core::NoneType start, int64_t stop,
                                      core::NoneType step) {
        return core::TensorKey::Slice(start, stop, step);
    });
    tensor_key.def_static("slice", [](core::NoneType start, core::NoneType stop,
                                      int64_t step) {
        return core::TensorKey::Slice(start, stop, step);
    });
    tensor_key.def_static("slice", [](core::NoneType start, core::NoneType stop,
                                      core::NoneType step) {
        return core::TensorKey::Slice(start, stop, step);
    });
    tensor_key.def_static("index_tensor", &core::TensorKey::IndexTensor);
}

}  // namespace open3d

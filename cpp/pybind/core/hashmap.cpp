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

#include "open3d/core/hashmap/Hashmap.h"

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {
void pybind_core_hashmap(py::module& m) {
    py::class_<Hashmap> hashmap(
            m, "Hashmap",
            "A Hashmap is a map from key to data wrapped by Tensors.");

    hashmap.def(py::init<size_t, const Dtype&, const Dtype&, const SizeVector&,
                         const SizeVector&, const Device&>(),
                "init_capacity"_a, "dtype_key"_a, "dtype_value"_a,
                "shape_key"_a, "shape_value"_a, "device"_a);

    hashmap.def("insert",
                [](Hashmap& h, const Tensor& keys, const Tensor& values) {
                    Tensor addrs, masks;
                    h.Insert(keys, values, addrs, masks);
                    return py::make_tuple(addrs, masks);
                });

    hashmap.def("activate", [](Hashmap& h, const Tensor& keys) {
        Tensor addrs, masks;
        h.Activate(keys, addrs, masks);
        return py::make_tuple(addrs, masks);
    });

    hashmap.def("find", [](Hashmap& h, const Tensor& keys) {
        Tensor addrs, masks;
        h.Find(keys, addrs, masks);
        return py::make_tuple(addrs, masks);
    });

    hashmap.def("erase", [](Hashmap& h, const Tensor& keys) {
        Tensor masks;
        h.Erase(keys, masks);
        return masks;
    });

    hashmap.def("get_active_addrs", [](Hashmap& h) {
        Tensor addrs;
        h.GetActiveIndices(addrs);
        return addrs;
    });

    hashmap.def("get_key_buffer", &Hashmap::GetKeyBuffer);
    hashmap.def("get_value_buffer", &Hashmap::GetValueBuffer);

    hashmap.def("get_key_tensor", &Hashmap::GetKeyTensor);
    hashmap.def("get_value_tensor", &Hashmap::GetValueTensor);

    hashmap.def("rehash", &Hashmap::Rehash);
    hashmap.def("size", &Hashmap::Size);
    hashmap.def("capacity", &Hashmap::GetCapacity);
}
}  // namespace core
}  // namespace open3d

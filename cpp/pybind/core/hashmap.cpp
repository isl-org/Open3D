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

    hashmap.def(py::init<size_t, const Dtype&, const Dtype&, const Device&>(),
                "init_capacity"_a, "dtype_key"_a, "dtype_val"_a, "device"_a);

    hashmap.def("insert",
                [](Hashmap& h, const Tensor& keys, const Tensor& values) {
                    Tensor iterators, masks;
                    h.Insert(keys, values, iterators, masks);
                    return py::make_tuple(iterators, masks);
                });

    hashmap.def("activate", [](Hashmap& h, const Tensor& keys) {
        Tensor iterators, masks;
        h.Activate(keys, iterators, masks);
        return py::make_tuple(iterators, masks);
    });

    hashmap.def("find", [](Hashmap& h, const Tensor& keys) {
        Tensor iterators, masks;
        h.Find(keys, iterators, masks);
        return py::make_tuple(iterators, masks);
    });

    hashmap.def("erase", [](Hashmap& h, const Tensor& keys) {
        Tensor masks;
        h.Erase(keys, masks);
        return masks;
    });

    hashmap.def("unpack_iterators", [](Hashmap& h, const Tensor& iterators,
                                       const Tensor& masks = Tensor()) {
        int64_t count = iterators.GetShape()[0];

        Tensor keys({count}, h.GetKeyDtype(), iterators.GetDevice());
        Tensor values({count}, h.GetValueDtype(), iterators.GetDevice());

        h.UnpackIterators(
                static_cast<const iterator_t*>(iterators.GetDataPtr()),
                static_cast<const bool*>(masks.GetDataPtr()), keys.GetDataPtr(),
                values.GetDataPtr(), count);

        return py::make_tuple(keys, values);
    });

    hashmap.def("assign_iterators", [](Hashmap& h, Tensor& iterators,
                                       Tensor& values,
                                       const Tensor& masks = Tensor()) {
        int64_t count = iterators.GetShape()[0];

        h.AssignIterators(static_cast<iterator_t*>(iterators.GetDataPtr()),
                          static_cast<const bool*>(masks.GetDataPtr()),
                          values.GetDataPtr(), count);

        return iterators;
    });

    hashmap.def("get_key_blob_as_tensor", &Hashmap::GetKeyBlobAsTensor);
    hashmap.def("get_value_blob_as_tensor", &Hashmap::GetValueBlobAsTensor);

    hashmap.def("rehash", &Hashmap::Rehash);
    hashmap.def("size", &Hashmap::Size);
    hashmap.def("capacity", &Hashmap::GetCapacity);
}
}  // namespace core
}  // namespace open3d

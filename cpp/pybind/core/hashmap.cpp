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

#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {

void pybind_core_hashmap(py::module& m) {
    py::class_<core::Hashmap> hashmap(
            m, "Hashmap",
            "A Hashmap is a map from key to data wrapped by Tensors.");

    hashmap.def(py::init([](size_t init_capacity, size_t bytesize_key,
                            size_t bytesize_value, const core::Device& device) {
                    return core::Hashmap(init_capacity, bytesize_key,
                                         bytesize_value, device);
                }),
                "init_capacity"_a, "bytesize_key"_a, "bytesize_value"_a,
                "device"_a);

    hashmap.def("insert", [](core::Hashmap& h, const core::Tensor& keys,
                             const core::Tensor& values) {
        int count = keys.GetShape()[0];
        core::Device device = keys.GetDevice();

        core::Tensor masks({count}, core::Dtype::Bool, device);
        void* iterators =
                core::MemoryManager::Malloc(sizeof(iterator_t) * count, device);

        h.Insert(keys.GetDataPtr(), values.GetDataPtr(),
                 static_cast<iterator_t*>(iterators),
                 static_cast<bool*>(masks.GetDataPtr()), count);

        core::MemoryManager::Free(iterators, device);
        return masks;
    });

    hashmap.def("activate", [](core::Hashmap& h, const core::Tensor& keys) {
        int count = keys.GetShape()[0];
        core::Device device = keys.GetDevice();

        core::Tensor masks({count}, core::Dtype::Bool, device);
        void* iterators =
                core::MemoryManager::Malloc(sizeof(iterator_t) * count, device);

        h.Activate(keys.GetDataPtr(), static_cast<iterator_t*>(iterators),
                   static_cast<bool*>(masks.GetDataPtr()), count);

        core::MemoryManager::Free(iterators, device);
        return masks;
    });

    hashmap.def("find", [](core::Hashmap& h, const core::Tensor& keys) {
        int count = keys.GetShape()[0];
        core::Device device = keys.GetDevice();

        core::Tensor masks({count}, core::Dtype::Bool, device);
        void* iterators =
                core::MemoryManager::Malloc(sizeof(iterator_t) * count, device);

        h.Find(keys.GetDataPtr(), static_cast<iterator_t*>(iterators),
               static_cast<bool*>(masks.GetDataPtr()), count);

        // TODO: unpack values to Tensors
        core::MemoryManager::Free(iterators, device);

        return masks;
    });

    hashmap.def("erase", [](core::Hashmap& h, const core::Tensor& keys) {
        int count = keys.GetShape()[0];
        core::Device device = keys.GetDevice();

        core::Tensor masks({count}, core::Dtype::Bool, device);
        void* iterators =
                core::MemoryManager::Malloc(sizeof(iterator_t) * count, device);

        h.Erase(keys.GetDataPtr(), static_cast<bool*>(masks.GetDataPtr()),
                count);

        core::MemoryManager::Free(iterators, device);
        return masks;
    });

    hashmap.def("rehash", &core::Hashmap::Rehash);
    hashmap.def("size", &core::Hashmap::Size);
}

}  // namespace open3d

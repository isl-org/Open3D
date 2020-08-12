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

#include "open3d/core/Device.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {

void pybind_core_device(py::module &m) {
    py::class_<core::Device> device(
            m, "Device",
            "Device context specifying device type and device id.");
    device.def(py::init<>())
            .def(py::init<core::Device::DeviceType, int>())
            .def(py::init<const std::string &, int>())
            .def(py::init<const std::string &>())
            .def("__eq__", &core::Device::operator==)
            .def("__ene__", &core::Device::operator!=)
            .def("__repr__", &core::Device::ToString)
            .def("__str__", &core::Device::ToString)
            .def("get_type", &core::Device::GetType)
            .def("get_id", &core::Device::GetID);

    py::enum_<core::Device::DeviceType>(device, "DeviceType")
            .value("CPU", core::Device::DeviceType::CPU)
            .value("CUDA", core::Device::DeviceType::CUDA)
            .export_values();
}
}  // namespace open3d

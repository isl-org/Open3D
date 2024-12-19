// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Device.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_device_declarations(py::module &m) {
    py::class_<Device> device(
            m, "Device",
            "Device context specifying device type and device id.");
    py::enum_<Device::DeviceType>(device, "DeviceType")
            .value("CPU", Device::DeviceType::CPU)
            .value("CUDA", Device::DeviceType::CUDA)
            .export_values();
}
void pybind_core_device_definitions(py::module &m) {
    auto device = static_cast<py::class_<Device>>(m.attr("Device"));
    device.def(py::init<>());
    device.def(py::init<Device::DeviceType, int>());
    device.def(py::init<const std::string &, int>());
    device.def(py::init<const std::string &>());
    device.def("__eq__", &Device::operator==);
    device.def("__ene__", &Device::operator!=);
    device.def("__repr__", &Device::ToString);
    device.def("__str__", &Device::ToString);
    device.def("get_type", &Device::GetType);
    device.def("get_id", &Device::GetID);
    device.def(py::pickle(
            [](const Device &d) {
                return py::make_tuple(d.GetType(), d.GetID());
            },
            [](py::tuple t) {
                if (t.size() != 2) {
                    utility::LogError(
                            "Cannot unpickle Device! Expecting a tuple of size "
                            "2.");
                }
                return Device(t[0].cast<Device::DeviceType>(),
                              t[1].cast<int>());
            }));
}

}  // namespace core
}  // namespace open3d

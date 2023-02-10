// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Optional.h"
#include "pybind/core/core.h"

namespace open3d {
namespace core {

void pybind_sycl_utils(py::module& m) { m.def("sycl_demo", &sycl::SYCLDemo); }

}  // namespace core
}  // namespace open3d

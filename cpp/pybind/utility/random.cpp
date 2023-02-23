// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Random.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {
namespace random {

void pybind_random(py::module &m) {
    py::module m_submodule = m.def_submodule("random");

    m_submodule.def("seed", &Seed, "seed"_a, "Set Open3D global random seed.");

    docstring::FunctionDocInject(m_submodule, "seed",
                                 {{"seed", "Random seed value."}});
}

}  // namespace random
}  // namespace utility
}  // namespace open3d

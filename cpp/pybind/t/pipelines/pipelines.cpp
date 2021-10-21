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

#include "pybind/t/pipelines/pipelines.h"

#include "pybind/open3d_pybind.h"
#include "pybind/t/pipelines/odometry/odometry.h"
#include "pybind/t/pipelines/registration/registration.h"
#include "pybind/t/pipelines/slac/slac.h"
#include "pybind/t/pipelines/slam/slam.h"

namespace open3d {
namespace t {
namespace pipelines {

void pybind_pipelines(py::module& m) {
    py::module m_pipelines = m.def_submodule(
            "pipelines", "Tensor-based geometry processing pipelines.");
    odometry::pybind_odometry(m_pipelines);
    registration::pybind_registration(m_pipelines);
    slac::pybind_slac(m_pipelines);
    slam::pybind_slam(m_pipelines);
}

}  // namespace pipelines
}  // namespace t
}  // namespace open3d

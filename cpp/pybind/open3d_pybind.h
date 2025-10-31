// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// If performing a debug build with VS2022 (_MSC_VER == 1930) we need to include
// corecrt.h before pybind so that the _STL_ASSERT macro is defined in a
// compatible way.
//
// pybind11/pybind11.h includes pybind11/detail/common.h, which undefines _DEBUG
// whilst including the Python headers (which in turn include corecrt.h). This
// alters how the _STL_ASSERT macro is defined and causes the build to fail.
//
// see https://github.com/microsoft/onnxruntime/issues/9735
//     https://github.com/microsoft/onnxruntime/pull/11495
//
#if defined(_MSC_FULL_VER) && defined(_DEBUG) && _MSC_FULL_VER >= 192930145
#include <corecrt.h>
#endif

#include <pybind11/detail/common.h>
#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include <optional>

#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/utility/Eigen.h"

// We include the type caster for tensor here because it must be included in
// every compilation unit.
#include "pybind/core/tensor_type_caster.h"

namespace py = pybind11;
using namespace py::literals;

typedef std::vector<Eigen::Matrix4d, open3d::utility::Matrix4d_allocator>
        temp_eigen_matrix4d;
typedef std::vector<Eigen::Vector4i, open3d::utility::Vector4i_allocator>
        temp_eigen_vector4i;

PYBIND11_MAKE_OPAQUE(std::vector<char>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix3d>);
PYBIND11_MAKE_OPAQUE(temp_eigen_matrix4d);
PYBIND11_MAKE_OPAQUE(temp_eigen_vector4i);
PYBIND11_MAKE_OPAQUE(
        std::vector<open3d::pipelines::registration::PoseGraphEdge>);
PYBIND11_MAKE_OPAQUE(
        std::vector<open3d::pipelines::registration::PoseGraphNode>);

namespace pybind11 {
namespace detail {

template <typename T, typename Class_>
void bind_default_constructor(Class_ &cl) {
    cl.def(py::init([]() { return new T(); }), "Default constructor");
}

template <typename T, typename Class_>
void bind_copy_functions(Class_ &cl) {
    cl.def(py::init([](const T &cp) { return new T(cp); }), "Copy constructor");
    cl.def("__copy__", [](T &v) { return T(v); });
    cl.def("__deepcopy__", [](T &v, py::dict &memo) { return T(v); });
}

}  // namespace detail
}  // namespace pybind11

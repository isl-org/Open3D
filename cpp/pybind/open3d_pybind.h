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
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Optional.h"

// We include the type caster for tensor here because it must be included in
// every compilation unit.
#include "pybind/core/tensor_type_caster.h"

// Replace with <pybind11/stl/filesystem.h> when we require C++17.
#include "pybind_filesystem.h"

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

/// Custom pybind11 type caster for open3d::utility::optional, which backports
/// C++17's `std::optional`. We need compiler supporting C++14 or newer.
/// Typically, this can be used to handle "None" parameters from Python.
///
/// Example python function:
///
/// ```python
/// def add(a, b=None):
///     # Assuming a, b are int.
///     if b is None:
///         return a
///     else:
///         return a + b
/// ```
///
/// Here's the equivalent C++ implementation:
///
/// ```cpp
/// m.def("add",
///     [](int a, open3d::utility::optional<int> b) {
///         if (!b.has_value()) {
///             return a;
///         } else {
///             return a + b.value();
///         }
///     },
///     py::arg("a"), py::arg("b") = py::none()
/// );
/// ```
///
/// Then this function can be called with:
///
/// ```python
/// add(1)
/// add(1, 2)
/// add(1, b=2)
/// add(1, b=None)
/// ```
template <typename T>
struct open3d_optional_caster {
    using value_conv = make_caster<typename T::value_type>;

    template <typename T_>
    static handle cast(T_ &&src, return_value_policy policy, handle parent) {
        if (!src) return none().inc_ref();
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<T>::policy(policy);
        }
        return value_conv::cast(*std::forward<T_>(src), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        } else if (src.is_none()) {
            return true;  // default-constructed value is already empty
        }
        value_conv inner_caster;
        if (!inner_caster.load(src, convert)) return false;

        value.emplace(
                cast_op<typename T::value_type &&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

template <typename T>
struct type_caster<open3d::utility::optional<T>>
    : public open3d_optional_caster<open3d::utility::optional<T>> {};

template <>
struct type_caster<open3d::utility::nullopt_t>
    : public void_caster<open3d::utility::nullopt_t> {};

}  // namespace detail
}  // namespace pybind11

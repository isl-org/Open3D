// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Adapted from <pybind11/stl/filesystem.h> to support C++14.
// Original attribution:
// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license.

#pragma once

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/detail/descr.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <string>

#ifdef _WIN32
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#ifdef __APPLE__
#include <filesystem>
namespace fs = std::__fs::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace pybind11 {
namespace detail {

template <typename T>
struct path_caster {
private:
    static PyObject *unicode_from_fs_native(const std::string &w) {
#if !defined(PYPY_VERSION)
        return PyUnicode_DecodeFSDefaultAndSize(w.c_str(), ssize_t(w.size()));
#else
        // PyPy mistakenly declares the first parameter as non-const.
        return PyUnicode_DecodeFSDefaultAndSize(const_cast<char *>(w.c_str()),
                                                ssize_t(w.size()));
#endif
    }

    static PyObject *unicode_from_fs_native(const std::wstring &w) {
        return PyUnicode_FromWideChar(w.c_str(), ssize_t(w.size()));
    }

public:
    static handle cast(const T &path, return_value_policy, handle) {
        if (auto py_str = unicode_from_fs_native(path.native())) {
            return module_::import("pathlib")
                    .attr("Path")(reinterpret_steal<object>(py_str))
                    .release();
        }
        return nullptr;
    }

    bool load(handle handle, bool) {
        // PyUnicode_FSConverter and PyUnicode_FSDecoder normally take care of
        // calling PyOS_FSPath themselves, but that's broken on PyPy (PyPy
        // issue #3168) so we do it ourselves instead.
        PyObject *buf = PyOS_FSPath(handle.ptr());
        if (!buf) {
            PyErr_Clear();
            return false;
        }
        PyObject *native = nullptr;
        if (std::is_same<typename T::value_type, char>::value) {
            if (PyUnicode_FSConverter(buf, &native) != 0) {
                if (auto *c_str = PyBytes_AsString(native)) {
                    // AsString returns a pointer to the internal buffer, which
                    // must not be free'd.
                    value = c_str;
                }
            }
        } else if (std::is_same<typename T::value_type, wchar_t>::value) {
            if (PyUnicode_FSDecoder(buf, &native) != 0) {
                if (auto *c_str = PyUnicode_AsWideCharString(native, nullptr)) {
                    // AsWideCharString returns a new string that must be
                    // free'd.
                    value = c_str;  // Copies the string.
                    PyMem_Free(c_str);
                }
            }
        }
        Py_XDECREF(native);
        Py_DECREF(buf);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    }

    PYBIND11_TYPE_CASTER(T,
                         io_name("Union[os.PathLike, str, bytes]",
                                 "pathlib.Path"));
};

template <>
struct type_caster<fs::path> : public path_caster<fs::path> {};

}  // namespace detail
}  // namespace pybind11

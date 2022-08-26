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

#pragma once

#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Optional.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

/// Convert Tensor to py::array (Numpy array). The python object holds a
/// reference to the Tensor, and when it goes out of scope, the Tensor's
/// reference counter will be decremented by 1.
///
/// You may use this helper function for exporting data to Numpy.
///
/// To expose a C++ buffer to python, we need to carefully manage the buffer
/// ownership. You first need to allocate the memory in the heap (e.g. with
/// `new`, `malloc`, avoid using containers that frees up memory when the C++
/// variable goes out of scope), then in pybind11, define a deleter function for
/// py::array_t that deallocates the buffer. This deleater function will be
/// called once the python reference count decreases to 0. See
/// https://stackoverflow.com/a/44682603/1255535 for details. This approach is
/// efficient since no memory copy is required.
///
/// Alternatively, you can create a Tensor with a **copy** of your data (so that
/// your original buffer can be freed), and let TensorToPyArray generate a
/// py::array that manages the buffer lifetime automatically. This is more
/// convenient, but will require an extra copy.
py::array TensorToPyArray(const Tensor& tensor);

/// Convert py::array (Numpy array) to Tensor.
///
/// You may use this helper function for importing data from Numpy.
///
/// \param inplace If True, Tensor will directly use the underlying Numpy
/// buffer. However, The data will become invalid once the Numpy variable is
/// deallocated, the Tensor's data becomes invalid without notice. If False, the
/// python buffer will be copied.
Tensor PyArrayToTensor(py::array array, bool inplace);

/// Convert py::list to Tensor.
///
/// Nested lists are supported, e.g. [[0, 1, 2], [3, 4, 5]] becomes a 2x3
/// tensor. For "ragged" list of invalid shapes, e.g. ((0, 1, 2, 3, 4, 5), (2,
/// 3)), the np_array's dtype is "O", a proper exception will be thrown.
///
/// The dtype will be inferred from the value of the list.
Tensor PyListToTensor(const py::list& list,
                      utility::optional<Dtype> dtype = utility::nullopt,
                      utility::optional<Device> device = utility::nullopt);

/// Convert py::tuple to Tensor.
///
/// Nested tuples are supported, e.g. ((0, 1, 2), (3, 4, 5)) becomes a 2x3
/// tensor. For "ragged" tuple of invalid shapes, e.g. ((0, 1, 2, 3, 4, 5), (2,
/// 3)), the np_array's dtype is "O", a proper exception will be thrown.
///
/// The dtype will be inferred from the value of the tuple.
Tensor PyTupleToTensor(const py::tuple& tuple,
                       utility::optional<Dtype> dtype = utility::nullopt,
                       utility::optional<Device> device = utility::nullopt);

/// Convert scalar double value to Tensor.
///
/// The default dtype is Float64, unless specified.
Tensor DoubleToTensor(double scalar_value,
                      utility::optional<Dtype> dtype = utility::nullopt,
                      utility::optional<Device> device = utility::nullopt);

/// Convert scalar int value to Tensor.
///
/// The default dtype is Int64, unless specified.
Tensor IntToTensor(int64_t scalar_value,
                   utility::optional<Dtype> dtype = utility::nullopt,
                   utility::optional<Device> device = utility::nullopt);

/// Convert scalar bool value to Tensor.
///
/// The default dtype is Bool, unless specified.
Tensor BoolToTensor(bool scalar_value,
                    utility::optional<Dtype> dtype = utility::nullopt,
                    utility::optional<Device> device = utility::nullopt);

/// Convert supported python types to Tensor.
///
/// Supported types:
/// 1) int
/// 2) float (double)
/// 3) list
/// 4) tuple
/// 5) numpy.ndarray (value will be copied)
/// 6) Tensor (value will be copied)
///
/// An exception will be thrown if the type is not supported.
Tensor PyHandleToTensor(const py::handle& handle,
                        utility::optional<Dtype> dtype = utility::nullopt,
                        utility::optional<Device> device = utility::nullopt,
                        bool force_copy = false);

}  // namespace core
}  // namespace open3d

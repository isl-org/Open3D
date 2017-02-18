// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "py3d.h"

namespace pybind11 {

template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
pybind11::class_<Vector, holder_type> bind_vector_without_repr(pybind11::module &m, std::string const &name, Args&&... args) {
	// hack function to disable __repr__ for the convenient function
	// bind_vector()
    using Class_ = pybind11::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(pybind11::init<>());
    detail::vector_if_copy_constructible<Vector, Class_>(cl);
    detail::vector_if_equal_operator<Vector, Class_>(cl);
    detail::vector_modifiers<Vector, Class_>(cl);
    detail::vector_accessor<Vector, Class_>(cl);
    cl.def("__bool__",
        [](const Vector &v) -> bool {
            return !v.empty();
        },
        "Check whether the list is nonempty"
    );
    cl.def("__len__", &Vector::size);
    return cl;
}

}	//namespace pybind11

void pybind_eigen(py::module &m)
{
	auto int_vector = py::bind_vector<std::vector<int>>(m, "IntVector",
			py::buffer_protocol());
	int_vector
		.def("__init__", [](std::vector<int> &v,
				py::array_t<int, py::array::c_style> b) {
			py::buffer_info info = b.request();
			if (info.format != py::format_descriptor<int>::format() || 
					info.ndim != 1)
				throw std::runtime_error("Incompatible buffer format!");
			v.resize(info.shape[0]);
			memcpy(v.data(), info.ptr, sizeof(int) * v.size());
		})
		.def_buffer([](std::vector<int> &v) -> py::buffer_info {
			return py::buffer_info(
					v.data(), sizeof(int), py::format_descriptor<int>::format(),
					1, {v.size()}, {sizeof(int)});
		});

	auto double_vector = 
			py::bind_vector<std::vector<double>>(m, "DoubleVector",
			py::buffer_protocol());
	double_vector
		.def("__init__", [](std::vector<double> &v,
				py::array_t<int, py::array::c_style> b) {
			py::buffer_info info = b.request();
			if (info.format != py::format_descriptor<double>::format() || 
					info.ndim != 1)
				throw std::runtime_error("Incompatible buffer format!");
			v.resize(info.shape[0]);
			memcpy(v.data(), info.ptr, sizeof(double) * v.size());
		})
		.def_buffer([](std::vector<double> &v) -> py::buffer_info {
			return py::buffer_info(
					v.data(), sizeof(double),
					py::format_descriptor<double>::format(),
					1, {v.size()}, {sizeof(double)});
		});

	auto vector3d_vector =
			py::bind_vector_without_repr<std::vector<Eigen::Vector3d>>(
			m, "Vector3dVector", py::buffer_protocol());
	vector3d_vector
		.def("__init__", [](std::vector<Eigen::Vector3d> &v,
				py::array_t<double, py::array::c_style> b) {
			py::buffer_info info = b.request();
			if (info.format != py::format_descriptor<double>::format() || 
					info.ndim != 2 || info.shape[1] != 3)
				throw std::runtime_error("Incompatible buffer format!");
			v.resize(info.shape[0]);
			memcpy(v.data(), info.ptr, sizeof(double) * 3 * v.size());
		})
		// Bare bones interface
		// We choose to disable them because they do not support ranged indices
		// such as [:,:].
		//.def("__getitem__", [](const std::vector<Eigen::Vector3d> &v,
		//		std::pair<size_t, size_t> i) {
		//	if (i.first >= v.size() || i.second >= 3)
		//		throw py::index_error();
		//	return v[i.first](i.second);
		//})
		//.def("__setitem__", [](std::vector<Eigen::Vector3d> &v,
		//		std::pair<size_t, size_t> i, double x) {
		//	if (i.first >= v.size() || i.second >= 3)
		//		throw py::index_error();
		//	v[i.first](i.second) = x;
		//})
		.def_buffer([](std::vector<Eigen::Vector3d> &v) -> py::buffer_info {
			return py::buffer_info(
					v.data(), sizeof(double),
					py::format_descriptor<double>::format(),
					2, {v.size(), 3}, {sizeof(double) * 3, sizeof(double)});
			})
		.def("__repr__", [](std::vector<Eigen::Vector3d> &v) {
			return std::string("std::vector of Eigen::Vector3d with ") +
					std::to_string(v.size()) + std::string(" elements.\n") +
					std::string("Use numpy.asarray() to access data.");
			});

	auto vector3i_vector =
			py::bind_vector_without_repr<std::vector<Eigen::Vector3i>>(
			m, "Vector3iVector", py::buffer_protocol());
	vector3i_vector
		.def("__init__", [](std::vector<Eigen::Vector3i> &v,
				py::array_t<int, py::array::c_style> b) {
			py::buffer_info info = b.request();
			if (info.format != py::format_descriptor<int>::format() || 
					info.ndim != 2 || info.shape[1] != 3)
				throw std::runtime_error("Incompatible buffer format!");
			v.resize(info.shape[0]);
			memcpy(v.data(), info.ptr, sizeof(int) * 3 * v.size());
		})
		// Bare bones interface
		// We choose to disable them because they do not support ranged indices
		// such as [:,:], see vector3d_vector for details.
		.def_buffer([](std::vector<Eigen::Vector3i> &v) -> py::buffer_info {
			return py::buffer_info(
					v.data(), sizeof(int),
					py::format_descriptor<int>::format(),
					2, {v.size(), 3}, {sizeof(int) * 3, sizeof(int)});
			})
		.def("__repr__", [](std::vector<Eigen::Vector3i> &v) {
			return std::string("std::vector of Eigen::Vector3i with ") +
					std::to_string(v.size()) + std::string(" elements.\n") +
					std::string("Use numpy.asarray() to access data.");
			});
}

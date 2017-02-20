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

namespace detail {

template <typename Vector, typename Class_>
void vector_modifiers_without_iterable(enable_if_t<std::is_copy_constructible<typename Vector::value_type>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;

    cl.def("append",
           [](Vector &v, const T &value) { v.push_back(value); },
           arg("x"),
           "Add an item to the end of the list");

    cl.def("extend",
       [](Vector &v, const Vector &src) {
           v.reserve(v.size() + src.size());
           v.insert(v.end(), src.begin(), src.end());
       },
       arg("L"),
       "Extend the list by appending all the items in the given list"
    );

    cl.def("insert",
        [](Vector &v, SizeType i, const T &x) {
            v.insert(v.begin() + (DiffType) i, x);
        },
        arg("i") , arg("x"),
        "Insert an item at a given position."
    );

    cl.def("pop",
        [](Vector &v) {
            if (v.empty())
                throw pybind11::index_error();
            T t = v.back();
            v.pop_back();
            return t;
        },
        "Remove and return the last item"
    );

    cl.def("pop",
        [](Vector &v, SizeType i) {
            if (i >= v.size())
                throw pybind11::index_error();
            T t = v[i];
            v.erase(v.begin() + (DiffType) i);
            return t;
        },
        arg("i"),
        "Remove and return the item at index ``i``"
    );

    cl.def("__setitem__",
        [](Vector &v, SizeType i, const T &t) {
            if (i >= v.size())
                throw pybind11::index_error();
            v[i] = t;
        }
    );

    /// Slicing protocol
    cl.def("__getitem__",
        [](const Vector &v, slice slice) -> Vector * {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            Vector *seq = new Vector();
            seq->reserve((size_t) slicelength);

            for (size_t i=0; i<slicelength; ++i) {
                seq->push_back(v[start]);
                start += step;
            }
            return seq;
        },
        arg("s"),
        "Retrieve list elements using a slice object"
    );

    cl.def("__setitem__",
        [](Vector &v, slice slice,  const Vector &value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            if (slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

            for (size_t i=0; i<slicelength; ++i) {
                v[start] = value[i];
                start += step;
            }
        },
        "Assign list elements using a slice object"
    );

    cl.def("__delitem__",
        [](Vector &v, SizeType i) {
            if (i >= v.size())
                throw pybind11::index_error();
            v.erase(v.begin() + DiffType(i));
        },
        "Delete the list elements at index ``i``"
    );

    cl.def("__delitem__",
        [](Vector &v, slice slice) {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            if (step == 1 && false) {
                v.erase(v.begin() + (DiffType) start, v.begin() + DiffType(start + slicelength));
            } else {
                for (size_t i = 0; i < slicelength; ++i) {
                    v.erase(v.begin() + DiffType(start));
                    start += step - 1;
                }
            }
        },
        "Delete list elements using a slice object"
    );
}

}

template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
pybind11::class_<Vector, holder_type> bind_vector_without_repr(pybind11::module &m, std::string const &name, Args&&... args) {
	// hack function to disable __repr__ for the convenient function
	// bind_vector()
    using Class_ = pybind11::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(pybind11::init<>());
    detail::vector_if_copy_constructible<Vector, Class_>(cl);
    detail::vector_if_equal_operator<Vector, Class_>(cl);
    detail::vector_modifiers_without_iterable<Vector, Class_>(cl);
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

namespace {

template <typename EigenVector>
void pybind_eigen_vector(py::module &m, const std::string &bind_name)
{
	typedef typename EigenVector::Scalar Scalar;
	py::class_<EigenVector> vec(m, bind_name.c_str(), py::buffer_protocol());
	vec.def("__init__", [](EigenVector &v,
			py::array_t<Scalar, py::array::c_style> b) {
		py::buffer_info info = b.request();
		if (info.format != py::format_descriptor<Scalar>::format() ||
				info.ndim != 1 ||
				info.shape[0] != EigenVector::RowsAtCompileTime)
			throw std::runtime_error("Incompatible buffer format!");
		new (&v) EigenVector();
		memcpy(v.data(), info.ptr, sizeof(EigenVector));
	});
	// Bare bones interface
	vec.def("__getitem__", [](EigenVector &v, size_t i) {
		if (i >= EigenVector::RowsAtCompileTime) throw py::index_error();
		return v(i);
	});
	vec.def("__setitem__", [](EigenVector &v, size_t i, Scalar x) {
		if (i >= EigenVector::RowsAtCompileTime) throw py::index_error();
		v(i) = x;
	});
	vec.def_buffer([](EigenVector &v) -> py::buffer_info {
		return py::buffer_info(
				v.data(), sizeof(Scalar),
				py::format_descriptor<Scalar>::format(),
				1, {EigenVector::RowsAtCompileTime}, {sizeof(Scalar)});
	});
	vec.def("__copy__", [](EigenVector &v) {
		return EigenVector(v);
	});
	vec.def("__deepcopy__", [](EigenVector &v, py::dict &memo) {
		return EigenVector(v);
	});
}

template <typename Scalar>
void pybind_eigen_vector_of_scalar(py::module &m, const std::string &bind_name)
{
	auto vec = py::bind_vector<std::vector<Scalar>>(m, bind_name,
			py::buffer_protocol());
	vec.def("__init__", [](std::vector<Scalar> &v,
			py::array_t<Scalar, py::array::c_style> b) {
		py::buffer_info info = b.request();
		if (info.format != py::format_descriptor<Scalar>::format() ||
				info.ndim != 1)
			throw std::runtime_error("Incompatible buffer format!");
		new (&v) std::vector<Scalar>(info.shape[0]);
		memcpy(v.data(), info.ptr, sizeof(Scalar) * v.size());
	});
	vec.def_buffer([](std::vector<Scalar> &v) -> py::buffer_info {
		return py::buffer_info(
				v.data(), sizeof(Scalar),
				py::format_descriptor<Scalar>::format(),
				1, {v.size()}, {sizeof(Scalar)});
	});
	vec.def("__copy__", [](std::vector<Scalar> &v) {
		return std::vector<Scalar>(v);
	});
	vec.def("__deepcopy__", [](std::vector<Scalar> &v, py::dict &memo) {
		return std::vector<Scalar>(v);
	});
}

template <typename EigenVector>
void pybind_eigen_vector_of_vector(py::module &m, const std::string &bind_name,
		const std::string &repr_name)
{
    typedef typename EigenVector::Scalar Scalar;
	auto vec = py::bind_vector_without_repr<std::vector<EigenVector>>(
			m, bind_name, py::buffer_protocol());
	vec.def("__init__", [](std::vector<EigenVector> &v,
			py::array_t<Scalar, py::array::c_style> b) {
		py::buffer_info info = b.request();
		if (info.format !=
				py::format_descriptor<Scalar>::format() ||
				info.ndim != 2 ||
				info.shape[1] != EigenVector::RowsAtCompileTime)
			throw std::runtime_error("Incompatible buffer format!");
		new (&v) std::vector<EigenVector>(info.shape[0]);
		memcpy(v.data(), info.ptr, sizeof(EigenVector) * v.size());
	});
	vec.def_buffer([](std::vector<EigenVector> &v) -> py::buffer_info {
		return py::buffer_info(
				v.data(), sizeof(Scalar),
				py::format_descriptor<Scalar>::format(),
				2, {v.size(), EigenVector::RowsAtCompileTime},
				{sizeof(EigenVector), sizeof(Scalar)});
	});
	vec.def("__repr__", [repr_name](std::vector<EigenVector> &v) {
		return repr_name + std::string(" with ") +
				std::to_string(v.size()) + std::string(" elements.\n") +
				std::string("Use numpy.asarray() to access data.");
	});
	vec.def("__copy__", [](std::vector<EigenVector> &v) {
		return std::vector<EigenVector>(v);
	});
	vec.def("__deepcopy__", [](std::vector<EigenVector> &v,
			py::dict &memo) {
		return std::vector<EigenVector>(v);
	});
	// Bare bones interface
	// We choose to disable them because they do not support slice indices
	// such as [:,:]. It is recommanded to convert it to numpy.asarray()
	// to access raw data.
	//v.def("__getitem__", [](const std::vector<Eigen::Vector3d> &v,
	//		std::pair<size_t, size_t> i) {
	//	if (i.first >= v.size() || i.second >= 3)
	//		throw py::index_error();
	//	return v[i.first](i.second);
	//});
	//v.def("__setitem__", [](std::vector<Eigen::Vector3d> &v,
	//		std::pair<size_t, size_t> i, double x) {
	//	if (i.first >= v.size() || i.second >= 3)
	//		throw py::index_error();
	//	v[i.first](i.second) = x;
	//});
}

}	// unnamed namespace

void pybind_eigen(py::module &m)
{
	pybind_eigen_vector<Eigen::Vector3d>(m, "Vector3d");
	pybind_eigen_vector<Eigen::Vector3i>(m, "Vector3i");
	pybind_eigen_vector_of_scalar<int>(m, "IntVector");
	pybind_eigen_vector_of_scalar<double>(m, "DoubleVector");

	pybind_eigen_vector_of_vector<Eigen::Vector3d>(m, "Vector3dVector",
			"std::vector<Eigen::Vector3d>");
	pybind_eigen_vector_of_vector<Eigen::Vector3i>(m, "Vector3iVector",
			"std::vector<Eigen::Vector3i>");
}

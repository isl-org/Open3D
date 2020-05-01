// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/open3d_pybind.h"

using namespace open3d;

namespace pybind11 {

template <typename Vector,
          typename holder_type = std::unique_ptr<Vector>,
          typename... Args>
py::class_<Vector, holder_type> bind_vector_without_repr(
        py::module &m, std::string const &name, Args &&... args) {
    // hack function to disable __repr__ for the convenient function
    // bind_vector()
    using Class_ = py::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(py::init<>());
    cl.def("__bool__", [](const Vector &v) -> bool { return !v.empty(); },
           "Check whether the list is nonempty");
    cl.def("__len__", &Vector::size);
    return cl;
}

// - This function is used by Pybind for std::vector<SomeEigenType> constructor.
//   This optional constructor is added to avoid too many Python <-> C++ API
//   calls when the vector size is large using the default biding method.
//   Pybind matches np.float64 array to py::array_t<double> buffer.
// - Directly using templates for the py::array_t<double> and py::array_t<int>
//   and etc. doesn't work. The current solution is to explicitly implement
//   bindings for each py array types.
template <typename EigenVector>
std::vector<EigenVector> py_array_to_vectors_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        // The EigenVector here must be a double-typed eigen vector, since only
        // open3d::Vector3dVector binds to py_array_to_vectors_double.
        // Therefore, we can use the memory map directly.
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector>
std::vector<EigenVector> py_array_to_vectors_int(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>>
std::vector<EigenVector, EigenAllocator>
py_array_to_vectors_int_eigen_allocator(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector, EigenAllocator> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>>
std::vector<EigenVector, EigenAllocator>
py_array_to_vectors_int64_eigen_allocator(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector, EigenAllocator> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

}  // namespace pybind11

namespace {

template <typename Scalar,
          typename Vector = std::vector<Scalar>,
          typename holder_type = std::unique_ptr<Vector>>
py::class_<Vector, holder_type> pybind_eigen_vector_of_scalar(
        py::module &m, const std::string &bind_name) {
    auto vec = py::bind_vector<std::vector<Scalar>>(m, bind_name,
                                                    py::buffer_protocol());
    vec.def_buffer([](std::vector<Scalar> &v) -> py::buffer_info {
        return py::buffer_info(v.data(), sizeof(Scalar),
                               py::format_descriptor<Scalar>::format(), 1,
                               {v.size()}, {sizeof(Scalar)});
    });
    vec.def("__copy__",
            [](std::vector<Scalar> &v) { return std::vector<Scalar>(v); });
    vec.def("__deepcopy__", [](std::vector<Scalar> &v, py::dict &memo) {
        return std::vector<Scalar>(v);
    });
    // We use iterable __init__ by default
    // vec.def("__init__", [](std::vector<Scalar> &v,
    //        py::array_t<Scalar, py::array::c_style> b) {
    //    py::buffer_info info = b.request();
    //    if (info.format != py::format_descriptor<Scalar>::format() ||
    //            info.ndim != 1)
    //        throw std::runtime_error("Incompatible buffer format!");
    //    new (&v) std::vector<Scalar>(info.shape[0]);
    //    memcpy(v.data(), info.ptr, sizeof(Scalar) * v.size());
    //});
    return vec;
}

template <typename EigenVector,
          typename Vector = std::vector<EigenVector>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_eigen_vector_of_vector(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name,
        InitFunc init_func) {
    typedef typename EigenVector::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<std::vector<EigenVector>>(
            m, bind_name, py::buffer_protocol());
    vec.def(py::init(init_func));
    vec.def_buffer([](std::vector<EigenVector> &v) -> py::buffer_info {
        size_t rows = EigenVector::RowsAtCompileTime;
        return py::buffer_info(v.data(), sizeof(Scalar),
                               py::format_descriptor<Scalar>::format(), 2,
                               {v.size(), rows},
                               {sizeof(EigenVector), sizeof(Scalar)});
    });
    vec.def("__repr__", [repr_name](const std::vector<EigenVector> &v) {
        return repr_name + std::string(" with ") + std::to_string(v.size()) +
               std::string(" elements.\n") +
               std::string("Use numpy.asarray() to access data.");
    });
    vec.def("__copy__", [](std::vector<EigenVector> &v) {
        return std::vector<EigenVector>(v);
    });
    vec.def("__deepcopy__", [](std::vector<EigenVector> &v, py::dict &memo) {
        return std::vector<EigenVector>(v);
    });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;

    // Bare bones interface
    // We choose to disable them because they do not support slice indices
    // such as [:,:]. It is recommended to convert it to numpy.asarray()
    // to access raw data.
    // v.def("__getitem__", [](const std::vector<Eigen::Vector3d> &v,
    //        std::pair<size_t, size_t> i) {
    //    if (i.first >= v.size() || i.second >= 3)
    //        throw py::index_error();
    //    return v[i.first](i.second);
    //});
    // v.def("__setitem__", [](std::vector<Eigen::Vector3d> &v,
    //        std::pair<size_t, size_t> i, double x) {
    //    if (i.first >= v.size() || i.second >= 3)
    //        throw py::index_error();
    //    v[i.first](i.second) = x;
    //});
    // We use iterable __init__ by default
    // vec.def("__init__", [](std::vector<EigenVector> &v,
    //        py::array_t<Scalar, py::array::c_style> b) {
    //    py::buffer_info info = b.request();s
    //    if (info.format !=
    //            py::format_descriptor<Scalar>::format() ||
    //            info.ndim != 2 ||
    //            info.shape[1] != EigenVector::RowsAtCompileTime)
    //        throw std::runtime_error("Incompatible buffer format!");
    //    new (&v) std::vector<EigenVector>(info.shape[0]);
    //    memcpy(v.data(), info.ptr, sizeof(EigenVector) * v.size());
    //});
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>,
          typename Vector = std::vector<EigenVector, EigenAllocator>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_eigen_vector_of_vector_eigen_allocator(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name,
        InitFunc init_func) {
    typedef typename EigenVector::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<
            std::vector<EigenVector, EigenAllocator>>(m, bind_name,
                                                      py::buffer_protocol());
    vec.def(py::init(init_func));
    vec.def_buffer(
            [](std::vector<EigenVector, EigenAllocator> &v) -> py::buffer_info {
                size_t rows = EigenVector::RowsAtCompileTime;
                return py::buffer_info(v.data(), sizeof(Scalar),
                                       py::format_descriptor<Scalar>::format(),
                                       2, {v.size(), rows},
                                       {sizeof(EigenVector), sizeof(Scalar)});
            });
    vec.def("__repr__",
            [repr_name](const std::vector<EigenVector, EigenAllocator> &v) {
                return repr_name + std::string(" with ") +
                       std::to_string(v.size()) + std::string(" elements.\n") +
                       std::string("Use numpy.asarray() to access data.");
            });
    vec.def("__copy__", [](std::vector<EigenVector, EigenAllocator> &v) {
        return std::vector<EigenVector, EigenAllocator>(v);
    });
    vec.def("__deepcopy__",
            [](std::vector<EigenVector, EigenAllocator> &v, py::dict &memo) {
                return std::vector<EigenVector, EigenAllocator>(v);
            });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

template <typename EigenMatrix,
          typename EigenAllocator = Eigen::aligned_allocator<EigenMatrix>,
          typename Vector = std::vector<EigenMatrix, EigenAllocator>,
          typename holder_type = std::unique_ptr<Vector>>
py::class_<Vector, holder_type> pybind_eigen_vector_of_matrix(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name) {
    typedef typename EigenMatrix::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<
            std::vector<EigenMatrix, EigenAllocator>>(m, bind_name,
                                                      py::buffer_protocol());
    vec.def_buffer(
            [](std::vector<EigenMatrix, EigenAllocator> &v) -> py::buffer_info {
                // We use this function to bind Eigen default matrix.
                // Thus they are all column major.
                size_t rows = EigenMatrix::RowsAtCompileTime;
                size_t cols = EigenMatrix::ColsAtCompileTime;
                return py::buffer_info(v.data(), sizeof(Scalar),
                                       py::format_descriptor<Scalar>::format(),
                                       3, {v.size(), rows, cols},
                                       {sizeof(EigenMatrix), sizeof(Scalar),
                                        sizeof(Scalar) * rows});
            });
    vec.def("__repr__",
            [repr_name](const std::vector<EigenMatrix, EigenAllocator> &v) {
                return repr_name + std::string(" with ") +
                       std::to_string(v.size()) + std::string(" elements.\n") +
                       std::string("Use numpy.asarray() to access data.");
            });
    vec.def("__copy__", [](std::vector<EigenMatrix, EigenAllocator> &v) {
        return std::vector<EigenMatrix, EigenAllocator>(v);
    });
    vec.def("__deepcopy__",
            [](std::vector<EigenMatrix, EigenAllocator> &v, py::dict &memo) {
                return std::vector<EigenMatrix, EigenAllocator>(v);
            });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

}  // unnamed namespace

void pybind_eigen(py::module &m) {
    auto intvector = pybind_eigen_vector_of_scalar<int>(m, "IntVector");
    intvector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert int32 numpy array of shape ``(n,)`` to Open3D format.)";
            }),
            py::none(), py::none(), "");

    auto doublevector =
            pybind_eigen_vector_of_scalar<double>(m, "DoubleVector");
    doublevector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert float64 numpy array of shape ``(n,)`` to Open3D format.)";
            }),
            py::none(), py::none(), "");

    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
            m, "Vector3dVector", "std::vector<Eigen::Vector3d>",
            py::py_array_to_vectors_double<Eigen::Vector3d>);
    vector3dvector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert float64 numpy array of shape ``(n, 3)`` to Open3D format.

Example usage

.. code-block:: python

    import open3d
    import numpy as np

    pcd = open3d.geometry.PointCloud()
    np_points = np.random.rand(100, 3)

    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(np_points)

    # From Open3D to numpy
    np_points = np.asarray(pcd.points)
)";
            }),
            py::none(), py::none(), "");

    auto vector3ivector = pybind_eigen_vector_of_vector<Eigen::Vector3i>(
            m, "Vector3iVector", "std::vector<Eigen::Vector3i>",
            py::py_array_to_vectors_int<Eigen::Vector3i>);
    vector3ivector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert int32 numpy array of shape ``(n, 3)`` to Open3D format..

Example usage

.. code-block:: python

    import open3d
    import numpy as np

    # Example mesh
    # x, y coordinates:
    # [0: (-1, 2)]__________[1: (1, 2)]
    #             \        /\
    #              \  (0) /  \
    #               \    / (1)\
    #                \  /      \
    #      [2: (0, 0)]\/________\[3: (2, 0)]
    #
    # z coordinate: 0

    mesh = open3d.geometry.TriangleMesh()
    np_vertices = np.array([[-1, 2, 0],
                            [1, 2, 0],
                            [0, 0, 0],
                            [2, 0, 0]])
    np_triangles = np.array([[0, 2, 1],
                             [1, 2, 3]]).astype(np.int32)
    mesh.vertices = open3d.Vector3dVector(np_vertices)

    # From numpy to Open3D
    mesh.triangles = open3d.Vector3iVector(np_triangles)

    # From Open3D to numpy
    np_triangles = np.asarray(mesh.triangles)
)";
            }),
            py::none(), py::none(), "");

    auto vector2ivector = pybind_eigen_vector_of_vector<Eigen::Vector2i>(
            m, "Vector2iVector", "std::vector<Eigen::Vector2i>",
            py::py_array_to_vectors_int<Eigen::Vector2i>);
    vector2ivector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert int32 numpy array of shape ``(n, 2)`` to "
                       "Open3D format.";
            }),
            py::none(), py::none(), "");

    auto vector2dvector = pybind_eigen_vector_of_vector<Eigen::Vector2d>(
            m, "Vector2dVector", "std::vector<Eigen::Vector2d>",
            py::py_array_to_vectors_double<Eigen::Vector2d>);
    vector2dvector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert float64 numpy array of shape ``(n, 2)`` to "
                       "Open3D format.";
            }),
            py::none(), py::none(), "");

    auto matrix4dvector = pybind_eigen_vector_of_matrix<Eigen::Matrix4d>(
            m, "Matrix4dVector", "std::vector<Eigen::Matrix4d>");
    matrix4dvector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert float64 numpy array of shape ``(n, 4, 4)`` to "
                       "Open3D format.";
            }),
            py::none(), py::none(), "");

    auto vector4ivector = pybind_eigen_vector_of_vector_eigen_allocator<
            Eigen::Vector4i>(
            m, "Vector4iVector", "std::vector<Eigen::Vector4i>",
            py::py_array_to_vectors_int_eigen_allocator<Eigen::Vector4i>);
    vector4ivector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert int numpy array of shape ``(n, 4)`` to "
                       "Open3D format.";
            }),
            py::none(), py::none(), "");
}

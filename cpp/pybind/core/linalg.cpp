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

#include "open3d/core/linalg/Inverse.h"
#include "open3d/core/linalg/LU.h"
#include "open3d/core/linalg/LeastSquares.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/core/linalg/SVD.h"
#include "open3d/core/linalg/Solve.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_linalg(py::module &m) {
    m.def(
            "matmul",
            [](const Tensor &A, const Tensor &B) {
                Tensor output;
                Matmul(A, B, output);
                return output;
            },
            "Function to perform matrix multiplication of two 2D tensors with "
            "compatible shapes.",
            "A"_a, "B"_a);

    m.def(
            "lu", [](const Tensor &A) { return A.LU(); },
            "Function to compute LU factorisation of a square 2D tensor.",
            "A"_a);

    m.def(
            "inv",
            [](const Tensor &A) {
                Tensor output;
                Inverse(A, output);
                return output;
            },
            "Function to inverse a square 2D tensor.", "A"_a);

    m.def(
            "solve",
            [](const Tensor &A, const Tensor &B) {
                Tensor output;
                Solve(A, B, output);
                return output;
            },
            "Function to solve X for a linear system AX = B where A is a "
            "square "
            "matrix",
            "A"_a, "B"_a);

    m.def(
            "lstsq",
            [](const Tensor &A, const Tensor &B) {
                Tensor output;
                LeastSquares(A, B, output);
                return output;
            },
            "Function to solve X for a linear system AX = B where A is a full "
            "rank matrix.",
            "A"_a, "B"_a);

    m.def(
            "svd",
            [](const Tensor &A) {
                Tensor U, S, VT;
                SVD(A, U, S, VT);
                return py::make_tuple(U, S, VT);
            },
            "Function to decompose A with A = U S VT.", "A"_a);
}

}  // namespace core
}  // namespace open3d

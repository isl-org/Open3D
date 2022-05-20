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

#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/t/pipelines/registration/registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

void pybind_robust_kernel_classes(py::module& m) {
    py::enum_<RobustKernelMethod>(m, "RobustKernelMethod",
                                  "Robust kernel method for outlier rejection.")
            .value("L2Loss", RobustKernelMethod::L2Loss)
            .value("L1Loss", RobustKernelMethod::L1Loss)
            .value("HuberLoss", RobustKernelMethod::HuberLoss)
            .value("CauchyLoss", RobustKernelMethod::CauchyLoss)
            .value("GMLoss", RobustKernelMethod::GMLoss)
            .value("TukeyLoss", RobustKernelMethod::TukeyLoss)
            .value("GeneralizedLoss", RobustKernelMethod::GeneralizedLoss)
            .export_values();

    // open3d.t.pipelines.odometry.OdometryConvergenceCriteria
    py::class_<RobustKernel> robust_kernel(m, "RobustKernel",
                                           R"(
Base class that models a robust kernel for outlier rejection. The virtual
function ``weight()`` must be implemented in derived classes.

The main idea of a robust loss is to downweight large residuals that are
assumed to be caused from outliers such that their influence on the solution
is reduced. This is achieved by optimizing:

.. math::
  \def\argmin{\mathop{\rm argmin}}
  \begin{equation}
    x^{*} = \argmin_{x} \sum_{i=1}^{N} \rho({r_i(x)})
  \end{equation}
  :label: robust_loss

where :math:`\rho(r)` is also called the robust loss or kernel and
:math:`r_i(x)` is the residual.

Several robust kernels have been proposed to deal with different kinds of
outliers such as Huber, Cauchy, and others.

The optimization problem in :eq:`robust_loss` can be solved using the
iteratively reweighted least squares (IRLS) approach, which solves a sequence
of weighted least squares problems. We can see the relation between the least
squares optimization in stanad non-linear least squares and robust loss
optimization by comparing the respective gradients which go to zero at the
optimum (illustrated only for the :math:`i^\mathrm{th}` residual):

.. math::
  \begin{eqnarray}
    \frac{1}{2}\frac{\partial (w_i r^2_i(x))}{\partial{x}}
    &=&
    w_i r_i(x) \frac{\partial r_i(x)}{\partial{x}} \\
    \label{eq:gradient_ls}
    \frac{\partial(\rho(r_i(x)))}{\partial{x}}
    &=&
    \rho'(r_i(x)) \frac{\partial r_i(x)}{\partial{x}}.
  \end{eqnarray}

By setting the weight :math:`w_i= \frac{1}{r_i(x)}\rho'(r_i(x))`, we
can solve the robust loss optimization problem by using the existing techniques
for weighted least-squares. This scheme allows standard solvers using
Gauss-Newton and Levenberg-Marquardt algorithms to optimize for robust losses
and is the one implemented in Open3D.

Then we minimize the objective function using Gauss-Newton and determine
increments by iteratively solving:

.. math::
  \newcommand{\mat}[1]{\mathbf{#1}}
  \newcommand{\veca}[1]{\vec{#1}}
  \renewcommand{\vec}[1]{\mathbf{#1}}
  \begin{align}
   \left(\mat{J}^\top \mat{W} \mat{J}\right)^{-1}\mat{J}^\top\mat{W}\vec{r},
  \end{align}

where :math:`\mat{W} \in \mathbb{R}^{n\times n}` is a diagonal matrix containing
weights :math:`w_i` for each residual :math:`r_i`

The different loss functions will only impact in the weight for each residual
during the optimization step.
Therefore, the only impact of the choice on the kernel is through its first
order derivate.

The kernels implemented so far, and the notation has been inspired by the
publication: **"Analysis of Robust Functions for Registration Algorithms"**, by
Philippe Babin et al.

For more information please also see: **"Adaptive Robust Kernels for
Non-Linear Least Squares Problems"**, by Nived Chebrolu et al.
)");
    py::detail::bind_copy_functions<RobustKernel>(robust_kernel);
    robust_kernel
            .def(py::init([](const RobustKernelMethod type,
                             const double scaling_parameter,
                             const double shape_parameter) {
                     return new RobustKernel(type, scaling_parameter,
                                             shape_parameter);
                 }),
                 "type"_a = RobustKernelMethod::L2Loss,
                 "scaling_parameter"_a = 1.0, "shape_parameter"_a = 1.0)
            .def_readwrite("type", &RobustKernel::type_, "Loss type.")
            .def_readwrite("scaling_parameter",
                           &RobustKernel::scaling_parameter_,
                           "Scaling parameter.")
            .def_readwrite("shape_parameter", &RobustKernel::shape_parameter_,
                           "Shape parameter.")
            .def("__repr__", [](const RobustKernel& rk) {
                return fmt::format(
                        "RobustKernel[scaling_parameter_={:e}, "
                        "shape_parameter_={:e}].",
                        rk.scaling_parameter_, rk.shape_parameter_);
            });
}

void pybind_robust_kernels(py::module& m) {
    py::module m_submodule = m.def_submodule(
            "robust_kernel",
            "Tensor-based robust kernel for outlier rejection.");
    pybind_robust_kernel_classes(m_submodule);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d

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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <memory>

#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pipelines/registration/registration.h"

namespace open3d {
namespace pipelines {
namespace registration {

template <class RobustKernelBase = RobustKernel>
class PyRobustKernelT : public RobustKernelBase {
public:
    using RobustKernelBase::RobustKernelBase;
    double Weight(double residual) const override {
        PYBIND11_OVERLOAD_PURE(double, RobustKernelBase, residual);
    }
};

// Type aliases to improve readability
using PyRobustKernel = PyRobustKernelT<RobustKernel>;
using PyL2Loss = PyRobustKernelT<L2Loss>;
using PyL1Loss = PyRobustKernelT<L1Loss>;
using PyHuberLoss = PyRobustKernelT<HuberLoss>;
using PyCauchyLoss = PyRobustKernelT<CauchyLoss>;
using PyGMLoss = PyRobustKernelT<GMLoss>;
using PyTukeyLoss = PyRobustKernelT<TukeyLoss>;

void pybind_robust_kernels(py::module &m) {
    // open3d.registration.RobustKernel
    py::class_<RobustKernel, std::shared_ptr<RobustKernel>, PyRobustKernel> rk(
            m, "RobustKernel",
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
    rk.def("weight", &RobustKernel::Weight, "residual"_a,
           "Obtain the weight for the given residual according to the "
           "robust kernel model.");
    docstring::ClassMethodDocInject(
            m, "RobustKernel", "weight",
            {{"residual", "value obtained during the optimization problem"}});

    // open3d.registration.L2Loss
    py::class_<L2Loss, std::shared_ptr<L2Loss>, PyL2Loss, RobustKernel> l2_loss(
            m, "L2Loss",
            R"(
The loss :math:`\rho(r)` for a given residual ``r`` is given by:

.. math:: \rho(r) = \frac{r^2}{2}

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math:: w(r) = 1
)");
    py::detail::bind_default_constructor<L2Loss>(l2_loss);
    py::detail::bind_copy_functions<L2Loss>(l2_loss);
    l2_loss.def("__repr__", [](const L2Loss &rk) {
        (void)rk;
        return "RobustKernel::L2Loss";
    });

    // open3d.registration.L1Loss:RobustKernel
    py::class_<L1Loss, std::shared_ptr<L1Loss>, PyL1Loss, RobustKernel> l1_loss(
            m, "L1Loss",
            R"(
The loss :math:`\rho(r)` for a given residual ``r`` is given by:

.. math:: \rho(r) = |r|

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math:: w(r) = \frac{1}{|r|}
)");
    py::detail::bind_default_constructor<L1Loss>(l1_loss);
    py::detail::bind_copy_functions<L1Loss>(l1_loss);
    l1_loss.def("__repr__", [](const L1Loss &rk) {
        (void)rk;
        return "RobustKernel::L1Loss";
    });

    // open3d.registration.HuberLoss
    py::class_<HuberLoss, std::shared_ptr<HuberLoss>, PyHuberLoss, RobustKernel>
            h_loss(m, "HuberLoss",
                   R"(
The loss :math:`\rho(r)` for a given residual ``r`` is:

.. math::
  \begin{equation}
    \rho(r)=
    \begin{cases}
      \frac{r^{2}}{2}, & |r| \leq k.\\
      k(|r|-k / 2), & \text{otherwise}.
    \end{cases}
  \end{equation}

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math::
  \begin{equation}
    w(r)=
    \begin{cases}
      1,              & |r| \leq k.       \\
      \frac{k}{|r|} , & \text{otherwise}.
    \end{cases}
  \end{equation}
)");
    py::detail::bind_copy_functions<HuberLoss>(h_loss);
    h_loss.def(py::init(
                       [](double k) { return std::make_shared<HuberLoss>(k); }),
               "k"_a)
            .def("__repr__",
                 [](const HuberLoss &rk) {
                     return std::string("RobustKernel::HuberLoss with k=") +
                            std::to_string(rk.k_);
                 })
            .def_readwrite("k", &HuberLoss::k_, "Parameter of the loss");

    // open3d.registration.CauchyLoss
    py::class_<CauchyLoss, std::shared_ptr<CauchyLoss>, PyCauchyLoss,
               RobustKernel>
            c_loss(m, "CauchyLoss",
                   R"(
The loss :math:`\rho(r)` for a given residual ``r`` is:

.. math::
  \begin{equation}
    \rho(r)=
    \frac{k^2}{2} \log\left(1 + \left(\frac{r}{k}\right)^2\right)
  \end{equation}

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math::
  \begin{equation}
    w(r)=
    \frac{1}{1 + \left(\frac{r}{k}\right)^2}
  \end{equation}
)");
    py::detail::bind_copy_functions<CauchyLoss>(c_loss);
    c_loss.def(py::init([](double k) {
                   return std::make_shared<CauchyLoss>(k);
               }),
               "k"_a)
            .def("__repr__",
                 [](const CauchyLoss &rk) {
                     return std::string("RobustKernel::CauchyLoss with k=") +
                            std::to_string(rk.k_);
                 })
            .def_readwrite("k", &CauchyLoss::k_, "Parameter of the loss.");

    // open3d.registration.GMLoss
    py::class_<GMLoss, std::shared_ptr<GMLoss>, PyGMLoss, RobustKernel> gm_loss(
            m, "GMLoss",
            R"(
The loss :math:`\rho(r)` for a given residual ``r`` is:

.. math::
  \begin{equation}
    \rho(r)=
    \frac{r^2/ 2}{k + r^2}
  \end{equation}

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math::
  \begin{equation}
    w(r)=
    \frac{k}{\left(k + r^2\right)^2}
  \end{equation}
)");
    py::detail::bind_copy_functions<GMLoss>(gm_loss);
    gm_loss.def(py::init([](double k) { return std::make_shared<GMLoss>(k); }),
                "k"_a)
            .def("__repr__",
                 [](const GMLoss &rk) {
                     return std::string("RobustKernel::GMLoss with k=") +
                            std::to_string(rk.k_);
                 })
            .def_readwrite("k", &GMLoss::k_, "Parameter of the loss.");

    // open3d.registration.TukeyLoss:RobustKernel
    py::class_<TukeyLoss, std::shared_ptr<TukeyLoss>, PyTukeyLoss, RobustKernel>
            t_loss(m, "TukeyLoss",
                   R"(
The loss :math:`\rho(r)` for a given residual ``r`` is:

.. math::
  \begin{equation}
    \rho(r)=
    \begin{cases}
      \frac{k^2\left[1-\left(1-\left(\frac{e}{k}\right)^2\right)^3\right]}{2}, & |r| \leq k.       \\
      \frac{k^2}{2},                                                           & \text{otherwise}.
    \end{cases}
  \end{equation}

The weight :math:`w(r)` for a given residual ``r`` is given by:

.. math::
  \begin{equation}
    w(r)=
    \begin{cases}
      \left(1 - \left(\frac{r}{k}\right)^2\right)^2, & |r| \leq k.       \\
      0 ,                                            & \text{otherwise}.
    \end{cases}
  \end{equation}
)");
    py::detail::bind_copy_functions<TukeyLoss>(t_loss);
    t_loss.def(py::init(
                       [](double k) { return std::make_shared<TukeyLoss>(k); }),
               "k"_a)
            .def("__repr__",
                 [](const TukeyLoss &tk) {
                     return std::string("RobustKernel::TukeyLoss with k=") +
                            std::to_string(tk.k_);
                 })
            .def_readwrite("k", &TukeyLoss::k_,
                           "``k`` Is a running constant for the loss.");
}  // namespace pipelines

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d

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

#include "open3d/utility/Random.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {
namespace random {

void pybind_random(py::module &m) {
    py::module m_submodule = m.def_submodule("random");

    m_submodule.def("seed", &Seed, "seed"_a, "Set Open3D global random seed.");

    docstring::FunctionDocInject(m_submodule, "seed",
                                 {{"seed", "Random seed value."}});

    // open3d::utility::random::UniformIntGenerator
    py::class_<UniformIntGenerator<int>> uniform_int_generator(
            m_submodule, "UniformIntGenerator",
            "Generate uniformly distributed integer values in [low, high). "
            "This class is globally seeded by open3d.utility.random.seed().");
    uniform_int_generator.def(py::init<int, int>(), "low"_a, "high"_a);
    uniform_int_generator.def("__call__", &UniformIntGenerator<int>::operator(),
                              "Generate a random number.");
    docstring::ClassMethodDocInject(
            m, "UniformIntGenerator", "__init__",
            {{"low", "Lower bound of the range (inclusive)."},
             {"high", "Upper bound of the range (exclusive)."}});
    docstring::ClassMethodDocInject(m, "UniformIntGenerator", "__call__");

    // open3d::utility::random::UniformRealGenerator
    py::class_<UniformRealGenerator<double>> uniform_real_generator(
            m_submodule, "UniformRealGenerator",
            "Generate uniformly distributed floating point values in [low, "
            "high). This class is globally seeded by "
            "open3d.utility.random.seed().");
    uniform_real_generator.def(py::init<double, double>(), "low"_a = 0.0,
                               "high"_a = 1.0);
    uniform_real_generator.def("__call__",
                               &UniformRealGenerator<double>::operator(),
                               "Generate a random number.");
    docstring::ClassMethodDocInject(
            m, "UniformRealGenerator", "__init__",
            {{"low", "Lower bound of the range (inclusive)."},
             {"high", "Upper bound of the range (exclusive)."}});
    docstring::ClassMethodDocInject(m, "UniformRealGenerator", "__call__");

    // open3d::utility::random::NormalGenerator
    py::class_<NormalGenerator<double>> normal_generator(
            m_submodule, "NormalGenerator",
            "Generate normally distributed floating point values with mean and "
            "standard deviation). This class is globally seeded by "
            "open3d.utility.random.seed().");
    normal_generator.def(py::init<double, double>(), "mean"_a = 0.0,
                         "stddev"_a = 1.0);
    normal_generator.def("__call__", &NormalGenerator<double>::operator(),
                         "Generate a random number.");
    docstring::ClassMethodDocInject(
            m_submodule, "NormalGenerator", "__init__",
            {{"mean", "Mean of the normal distribution."},
             {"stddev", "Standard deviation of the normal distribution."}});
    docstring::ClassMethodDocInject(m_submodule, "NormalGenerator", "__call__");
}

}  // namespace random
}  // namespace utility
}  // namespace open3d

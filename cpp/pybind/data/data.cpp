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

#include "pybind/data/data.h"

#include "open3d/data/Dataset.h"
#include "pybind/data/data_trampoline.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace data {

void pybind_data_classes(py::module& m) {
    // open3d.data.Dataset
    py::class_<Dataset, PyDataset<Dataset>, std::shared_ptr<Dataset>> dataset(
            m, "Dataset", "The base dataset class.");
    dataset.def(py::init<const std::string&, const std::string&,
                         const std::string&>(),
                "prefix"_a, "help"_a = "", "data_root"_a = "")
            .def_property_readonly("data_root", &Dataset::GetDataRoot,
                                   "Returns data root path.")
            .def_property_readonly("prefix", &Dataset::GetPrefix,
                                   "Returns data prefix.")
            .def_property_readonly("help_string", &Dataset::GetHelpString,
                                   "Returns string of helpful documentation.")
            .def("extract_dir", &Dataset::GetExtractDir,
                 "Returns data extract directory path.",
                 "relative_to_data_root"_a = false)
            .def("download_dir", &Dataset::GetDownloadDir,
                 "Returns data download directory path.",
                 "relative_to_data_root"_a = false);
    docstring::ClassMethodDocInject(m, "Dataset", "data_root");
    docstring::ClassMethodDocInject(m, "Dataset", "prefix");
    docstring::ClassMethodDocInject(m, "Dataset", "help_string");
    docstring::ClassMethodDocInject(m, "Dataset", "extract_dir");
    docstring::ClassMethodDocInject(m, "Dataset", "download_dir");

    // open3d.data.SimpleDataset
    py::class_<SimpleDataset, PySimpleDataset<SimpleDataset>,
               std::shared_ptr<SimpleDataset>, Dataset>
            simple_dataset(m, "SimpleDataset", "Simple dataset class.");
    simple_dataset.def(
            py::init<const std::string&, const std::vector<std::string>&,
                     const std::string&, const bool, const std::string&,
                     const std::string&>(),
            "prefix"_a, "urls"_a, "md5"_a, "no_extract"_a = false,
            "help_string"_a = "", "data_root"_a = "");
}

void pybind_data(py::module& m) {
    py::module m_submodule = m.def_submodule("data", "Data handling module.");
    pybind_data_classes(m_submodule);
    dataset::pybind_dataset(m_submodule);
}

}  // namespace data
}  // namespace open3d

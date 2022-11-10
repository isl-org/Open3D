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

#include "pybind/data/dataset.h"

#include "open3d/data/Dataset.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace data {

template <class DatasetBase = Dataset>
class PyDataset : public DatasetBase {
public:
    using DatasetBase::DatasetBase;
};

template <class DownloadDatasetBase = DownloadDataset>
class PyDownloadDataset : public PyDataset<DownloadDatasetBase> {
public:
    using PyDataset<DownloadDatasetBase>::PyDataset;
};

void pybind_data_classes(py::module& m) {
    // open3d.data.open3d_downloads_prefix as static attr of open3d.data.
    m.attr("open3d_downloads_prefix") = py::cast(Open3DDownloadsPrefix());

    // open3d.data.DataDescriptor
    py::class_<DataDescriptor> data_descriptor(
            m, "DataDescriptor",
            "DataDescriptor is a class that describes a data file. It contains "
            "the URL mirrors to download the file, the MD5 hash of the file, "
            "and wether to extract the file.");
    data_descriptor
            .def(py::init([](const std::vector<std::string>& urls,
                             const std::string& md5,
                             const std::string& extract_in_subdir) {
                     return DataDescriptor{urls, md5, extract_in_subdir};
                 }),
                 "urls"_a, "md5"_a, "extract_in_subdir"_a = "")
            .def(py::init([](const std::string& url, const std::string& md5,
                             const std::string& extract_in_subdir) {
                     return DataDescriptor{std::vector<std::string>{url}, md5,
                                           extract_in_subdir};
                 }),
                 "url"_a, "md5"_a, "extract_in_subdir"_a = "")
            .def_readonly("urls", &DataDescriptor::urls_,
                          "URL to download the data file.")
            .def_readonly("md5", &DataDescriptor::md5_,
                          "MD5 hash of the data file.")
            .def_readonly("extract_in_subdir",
                          &DataDescriptor::extract_in_subdir_,
                          "Subdirectory to extract the file. If empty, the "
                          "file will be extracted in the root extract "
                          "directory of the dataset.");

    // open3d.data.Dataset
    py::class_<Dataset, PyDataset<Dataset>, std::shared_ptr<Dataset>> dataset(
            m, "Dataset", "The base dataset class.");
    dataset.def(py::init<const std::string&, const std::string&>(), "prefix"_a,
                "data_root"_a = "")
            .def_property_readonly(
                    "data_root", &Dataset::GetDataRoot,
                    "Get data root directory. The data root is set at "
                    "construction time or automatically determined.")
            .def_property_readonly("prefix", &Dataset::GetPrefix,
                                   "Get prefix for the dataset.")
            .def_property_readonly(
                    "download_dir", &Dataset::GetDownloadDir,
                    "Get absolute path to download directory. i.e. "
                    "${data_root}/${download_prefix}/${prefix}")
            .def_property_readonly(
                    "extract_dir", &Dataset::GetExtractDir,
                    "Get absolute path to extract directory. i.e. "
                    "${data_root}/${extract_prefix}/${prefix}");
    docstring::ClassMethodDocInject(m, "Dataset", "data_root");
    docstring::ClassMethodDocInject(m, "Dataset", "prefix");
    docstring::ClassMethodDocInject(m, "Dataset", "download_dir");
    docstring::ClassMethodDocInject(m, "Dataset", "extract_dir");

    // open3d.data.DownloadDataset
    py::class_<DownloadDataset, PyDownloadDataset<DownloadDataset>,
               std::shared_ptr<DownloadDataset>, Dataset>
            single_download_dataset(m, "DownloadDataset",
                                    "Single file download dataset class.");
    single_download_dataset.def(
            py::init<const std::string&, const DataDescriptor&,
                     const std::string&>(),
            "prefix"_a, "data_descriptor"_a, "data_root"_a = "");
}

void pybind_armadillo(py::module& m) {
    // open3d.data.ArmadilloMesh
    py::class_<ArmadilloMesh, PyDownloadDataset<ArmadilloMesh>,
               std::shared_ptr<ArmadilloMesh>, DownloadDataset>
            armadillo(m, "ArmadilloMesh",
                      "Data class for `ArmadilloMesh` contains the "
                      "`ArmadilloMesh.ply` from the `Stanford 3D Scanning "
                      "Repository`.");
    armadillo.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("path", &ArmadilloMesh::GetPath,
                                   "Path to the `ArmadilloMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "ArmadilloMesh", "path");
}

void pybind_data(py::module& m) {
    py::module m_submodule = m.def_submodule("data", "Data handling module.");
    pybind_data_classes(m_submodule);
    // Demo data.

    pybind_armadillo(m_submodule);
}

}  // namespace data
}  // namespace open3d

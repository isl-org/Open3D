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

#include "open3d/data/Dataset.h"

#include <string>

#include "pybind/data/data.h"
#include "pybind/data/data_trampoline.h"
#include "pybind/docstring.h"

namespace open3d {
namespace data {
namespace dataset {

void pybind_sample_icp_pointclouds(py::module& m) {
    py::class_<SampleICPPointClouds, PySimpleDataset<SampleICPPointClouds>,
               std::shared_ptr<SampleICPPointClouds>, SimpleDataset>
            sample_icp_pointclouds(m, "SampleICPPointClouds",
                                   "SampleICPPointClouds dataset.");
    sample_icp_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleICPPointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const SampleICPPointClouds& sample_icp_pointclouds) {
                        return sample_icp_pointclouds.GetPaths();
                    },
                    "List of path to point-cloud fragments. "
                    "paths[x] returns path to `cloud_bin_x.pcd` point-cloud "
                    "where x is from 0 to 2.");
    docstring::ClassMethodDocInject(m, "SampleICPPointClouds", "paths");
}

void pybind_redwood_living_room_fragments(py::module& m) {
    py::class_<RedwoodLivingRoomFragments,
               PySimpleDataset<RedwoodLivingRoomFragments>,
               std::shared_ptr<RedwoodLivingRoomFragments>, SimpleDataset>
            redwood_living_room_fragments(
                    m, "RedwoodLivingRoomFragments",
                    "RedwoodLivingRoomFragments dataset.");
    redwood_living_room_fragments
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodLivingRoomFragments", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodLivingRoomFragments&
                               redwood_living_room_fragments) {
                        return redwood_living_room_fragments.GetPaths();
                    },
                    "List of path to point-cloud fragments. "
                    "paths[x] returns path to `cloud_bin_x.ply` point-cloud "
                    "where x is from 0 to 56.");
    docstring::ClassMethodDocInject(m, "RedwoodLivingRoomFragments", "paths");
}

void pybind_redwood_office_fragments(py::module& m) {
    py::class_<RedwoodOfficeFragments, PySimpleDataset<RedwoodOfficeFragments>,
               std::shared_ptr<RedwoodOfficeFragments>, SimpleDataset>
            redwood_office_fragments(m, "RedwoodOfficeFragments",
                                     "RedwoodOfficeFragments dataset.");
    redwood_office_fragments
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodOfficeFragments", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodOfficeFragments& redwood_office_fragments) {
                        return redwood_office_fragments.GetPaths();
                    },
                    "List of path to point-cloud fragments. "
                    "paths[x] returns path to `cloud_bin_x.ply` point-cloud "
                    "where x is from 0 to 51.");
    docstring::ClassMethodDocInject(m, "RedwoodOfficeFragments", "paths");
}

void pybind_dataset(py::module& m) {
    py::module m_dataset = m.def_submodule("dataset", "Dataset module.");
    pybind_sample_icp_pointclouds(m_dataset);
    pybind_redwood_living_room_fragments(m_dataset);
    pybind_redwood_office_fragments(m_dataset);
}

}  // namespace dataset
}  // namespace data
}  // namespace open3d

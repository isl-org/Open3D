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

void pybind_sample_pcd_fragments(py::module& m) {
    py::class_<SamplePCDFragments, PySimpleDataset<SamplePCDFragments>,
               std::shared_ptr<SamplePCDFragments>, SimpleDataset>
            sample_pcd_fragments(m, "SamplePCDFragments",
                                 "SamplePCDFragments dataset.");
    sample_pcd_fragments.def(py::init<const std::string&, const std::string&>(),
                             "prefix"_a = "SamplePCDFragments",
                             "data_root"_a = "");
    //         .def_readwrite("path_to_fragments",
    //                        &SamplePCDFragments::path_to_fragments_,
    //                        "List of path to PCD point-cloud fragments. "
    //                        "path_to_fragments_[x] returns path to "
    //                        "`cloud_bin_x.pcd` where x is from 0 to 2.");
    // docstring::ClassMethodDocInject(m, "SamplePCDFragments",
    //                                 "path_to_fragments");
}

void pybind_redwood_living_room_fragments(py::module& m) {
    py::class_<RedwoodLivingRoomFragments,
               PySimpleDataset<RedwoodLivingRoomFragments>,
               std::shared_ptr<RedwoodLivingRoomFragments>, SimpleDataset>
            redwood_living_room_fragments(
                    m, "RedwoodLivingRoomFragments",
                    "RedwoodLivingRoomFragments dataset.");
    //     // py::detail::bind_default_constructor<RedwoodLivingRoomFragments>(
    //     //             redwood_living_room_fragments);
    //     //     py::detail::bind_copy_functions<RedwoodLivingRoomFragments>(
    //     //             redwood_living_room_fragments);

    redwood_living_room_fragments.def(
            py::init<const std::string&, const std::string&>(),
            "prefix"_a = "RedwoodLivingRoomFragments", "data_root"_a = "");
    //             .def_readwrite("path_to_fragments",
    //                            &RedwoodLivingRoomFragments::path_to_fragments_,
    //                            "List of path to PCD point-cloud fragments. "
    //                            "path_to_fragments_[x] returns path to "
    //                            "`cloud_bin_x.pcd` where x is from 0 to 56.");
    //     docstring::ClassMethodDocInject(m, "RedwoodLivingRoomFragments",
    //                                     "path_to_fragments");
}

void pybind_redwood_office_fragments(py::module& m) {
    py::class_<RedwoodOfficeFragments, PySimpleDataset<RedwoodOfficeFragments>,
               std::shared_ptr<RedwoodOfficeFragments>, SimpleDataset>
            redwood_office_fragments(m, "RedwoodOfficeFragments",
                                     "RedwoodOfficeFragments dataset.");
    //     //     py::detail::bind_default_constructor<RedwoodOfficeFragments>(
    //     //             redwood_office_fragments);
    //     //     py::detail::bind_copy_functions<RedwoodOfficeFragments>(
    //     //             redwood_office_fragments);

    redwood_office_fragments.def(
            py::init<const std::string&, const std::string&>(),
            "prefix"_a = "RedwoodOfficeFragments", "data_root"_a = "");
    //             .def_readwrite("path_to_fragments",
    //                            &RedwoodOfficeFragments::path_to_fragments_,
    //                            "List of path to PCD point-cloud fragments. "
    //                            "path_to_fragments_[x] returns path to "
    //                            "`cloud_bin_x.pcd` where x is from 0 to 51.");
    //     docstring::ClassMethodDocInject(m, "RedwoodOfficeFragments",
    //                                     "path_to_fragments");
}

void pybind_dataset(py::module& m) {
    py::module m_dataset = m.def_submodule("dataset", "Dataset module.");
    pybind_sample_pcd_fragments(m_dataset);
    pybind_redwood_living_room_fragments(m_dataset);
    pybind_redwood_office_fragments(m_dataset);
}

}  // namespace dataset
}  // namespace data
}  // namespace open3d

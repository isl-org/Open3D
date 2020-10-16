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

#include <rply.h>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

namespace ply_pointcloud_reader {

struct PLYReaderState {
    utility::CountingProgressReporter *progress_bar;
    std::unordered_map<std::string, core::TensorList> attributes;
    std::vector<std::string> attribute_name;
    std::vector<int64_t> attribute_index;
    std::vector<int64_t> attribute_num;
};

int ReadPropertyCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long id;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &id);
    if (state_ptr->attribute_index[id] >= state_ptr->attribute_num[id]) {
        return 0;
    }

    double value = ply_get_argument_value(argument);

    state_ptr->attributes[state_ptr->attribute_name[id]]
                         [state_ptr->attribute_index[id]][0] = value;
    state_ptr->attribute_index[id]++;
    if (state_ptr->attribute_index[id] % 1000 == 0) {
        state_ptr->progress_bar->Update(state_ptr->attribute_index[id]);
    }
    return 1;
}

bool ConcatColumns(core::TensorList a,
                   core::TensorList b,
                   core::TensorList c,
                   core::TensorList &combined) {
    if (!(a.GetSize() == b.GetSize()) || !(a.GetSize() == c.GetSize()))
        return false;
    if (!(a.GetDtype() == b.GetDtype()) || !(a.GetDtype() == c.GetDtype()))
        return false;

    combined = core::TensorList(a.GetSize(), {3}, a.GetDtype());

    combined.AsTensor().Slice(1, 0, 1) = a.AsTensor();
    combined.AsTensor().Slice(1, 1, 2) = b.AsTensor();
    combined.AsTensor().Slice(1, 2, 3) = c.AsTensor();

    return true;
}

core::Dtype getDtype(e_ply_type type) {
    // PLY_LIST attribute is not supported.
    // Currently, we are not doing datatype conversions, so some of the ply
    // datatypes are not included.

    if (type == PLY_UINT8) {
        return core::Dtype::UInt8;
    } else if (type == PLY_UINT16) {
        return core::Dtype::UInt16;
    } else if (type == PLY_INT32) {
        return core::Dtype::Int32;
    } else if (type == PLY_FLOAT32) {
        return core::Dtype::Float32;
    } else if (type == PLY_FLOAT64) {
        return core::Dtype::Float64;
    } else if (type == PLY_UCHAR) {
        return core::Dtype::UInt8;
    } else if (type == PLY_INT) {
        return core::Dtype::Int32;
    } else if (type == PLY_FLOAT) {
        return core::Dtype::Float32;
    } else if (type == PLY_DOUBLE) {
        return core::Dtype::Float64;
    }

    return core::Dtype::Undefined;
}

}  // namespace ply_pointcloud_reader

bool ReadPointCloudFromPLY(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const open3d::io::ReadPointCloudOption &params) {
    using namespace ply_pointcloud_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    p_ply_property property = NULL;
    e_ply_type type, length_type, value_type;
    long property_id = 0;
    const char *property_nm;

    // Get first ply element; assuming it will be vertex.
    p_ply_element element = ply_get_next_element(ply_file, NULL);
    property = ply_get_next_property(element, NULL);

    while (property) {
        ply_get_property_info(property, &property_nm, &type, &length_type,
                              &value_type);

        if (getDtype(type) == core::Dtype::Undefined) {
            utility::LogWarning("Read PLY failed: unsupported datatype");
            ply_close(ply_file);
            return false;
        }

        state.attribute_name.push_back(property_nm);
        state.attribute_num.push_back(ply_set_read_cb(
                ply_file, "vertex", state.attribute_name[property_id].c_str(),
                ReadPropertyCallback, &state, property_id));

        state.attribute_index.push_back(0);
        state.attributes[state.attribute_name[property_id]] = core::TensorList(
                state.attribute_num[property_id], {1}, getDtype(type));
        // Get next property.
        property = ply_get_next_property(element, property);
        property_id++;
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(state.attribute_num[0]);
    state.progress_bar = &reporter;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}",
                            filename);
        ply_close(ply_file);
        return false;
    }

    pointcloud.Clear();

    // Add base attributes.
    if (state.attributes.find("x") != state.attributes.end()) {
        core::TensorList points;
        if (ConcatColumns(state.attributes["x"], state.attributes["y"],
                          state.attributes["z"], points)) {
            state.attributes.erase("x");
            state.attributes.erase("y");
            state.attributes.erase("z");
            pointcloud.SetPoints(points);
        }
    }
    if (state.attributes.find("nx") != state.attributes.end()) {
        core::TensorList normals;
        if (ConcatColumns(state.attributes["nx"], state.attributes["ny"],
                          state.attributes["nz"], normals)) {
            state.attributes.erase("nx");
            state.attributes.erase("ny");
            state.attributes.erase("nz");
            pointcloud.SetPointNormals(normals);
        }
    }
    if (state.attributes.find("red") != state.attributes.end()) {
        core::TensorList colors;
        if (ConcatColumns(state.attributes["red"], state.attributes["green"],
                          state.attributes["blue"], colors)) {
            state.attributes.erase("red");
            state.attributes.erase("green");
            state.attributes.erase("blue");
            pointcloud.SetPointColors(colors);
        }
    }

    // Add rest of the attributes.
    for (size_t i = 0; i < state.attributes.size(); i++) {
        pointcloud.SetPointAttr(state.attribute_name[i],
                                state.attributes[state.attribute_name[i]]);
    }

    ply_close(ply_file);
    reporter.Finish();

    return true;
}

}  // namespace io

}  // namespace t

}  // namespace open3d
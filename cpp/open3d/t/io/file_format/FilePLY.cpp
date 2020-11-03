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

struct PLYReaderState {
    struct AttrState {
        std::string name_;
        int64_t size_;
        core::Tensor data_;
        int64_t current_size_;
    };
    // Allow fast access of attr_state by name.
    std::unordered_map<std::string, std::shared_ptr<AttrState>>
            name_to_attr_state_;
    // Allow fast access of attr_state by index.
    std::vector<std::shared_ptr<AttrState>> id_to_attr_state_;
    utility::CountingProgressReporter *progress_bar_;
};

template <typename T>
static int ReadAttributeCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long id;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &id);
    std::shared_ptr<PLYReaderState::AttrState> &attr_state =
            state_ptr->id_to_attr_state_[id];
    if (attr_state->current_size_ >= attr_state->size_) {
        return 0;
    }

    T *data_ptr = static_cast<T *>(attr_state->data_.GetDataPtr());
    data_ptr[attr_state->current_size_++] =
            static_cast<T>(ply_get_argument_value(argument));

    if (attr_state->current_size_ % 1000 == 0) {
        state_ptr->progress_bar_->Update(attr_state->current_size_);
    }
    return 1;
}

static core::TensorList ConcatColumns(const core::Tensor &a,
                                      const core::Tensor &b,
                                      const core::Tensor &c) {
    if (a.NumDims() != 1 || b.NumDims() != 1 || c.NumDims() != 1) {
        utility::LogError("Read PLY failed: only 1D attributes are supported.");
    }

    if ((a.GetShape()[0] != b.GetShape()[0]) ||
        (a.GetShape()[0] != c.GetShape()[0])) {
        utility::LogError("Read PLY failed: size mismatch in base attributes.");
    }
    if ((a.GetDtype() != b.GetDtype()) || (a.GetDtype() != c.GetDtype())) {
        utility::LogError(
                "Read PLY failed: datatype mismatch in base attributes.");
    }

    core::TensorList combined =
            core::TensorList(a.GetShape()[0], {3}, a.GetDtype());
    combined.AsTensor().IndexExtract(1, 0) = a;
    combined.AsTensor().IndexExtract(1, 1) = b;
    combined.AsTensor().IndexExtract(1, 2) = c;

    return combined;
}

// Some of these datatypes are supported by TensorList but are added here just
// for completeness.
static std::string GetDtypeString(e_ply_type type) {
    if (type == PLY_INT8) {
        return "int8";
    } else if (type == PLY_UINT8) {
        return "uint8";
    } else if (type == PLY_INT16) {
        return "int16";
    } else if (type == PLY_UINT16) {
        return "uint16";
    } else if (type == PLY_INT32) {
        return "int32";
    } else if (type == PLY_UIN32) {
        return "uint32";
    } else if (type == PLY_FLOAT32) {
        return "float32";
    } else if (type == PLY_FLOAT64) {
        return "float64";
    } else if (type == PLY_CHAR) {
        return "char";
    } else if (type == PLY_UCHAR) {
        return "uchar";
    } else if (type == PLY_SHORT) {
        return "short";
    } else if (type == PLY_USHORT) {
        return "ushort";
    } else if (type == PLY_INT) {
        return "int";
    } else if (type == PLY_UINT) {
        return "uint";
    } else if (type == PLY_FLOAT) {
        return "float";
    } else if (type == PLY_DOUBLE) {
        return "double";
    } else if (type == PLY_LIST) {
        return "list";
    } else {
        return "unknown";
    }
}

static core::Dtype GetDtype(e_ply_type type) {
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
    } else {
        return core::Dtype::Undefined;
    }
}

bool ReadPointCloudFromPLY(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const open3d::io::ReadPointCloudOption &params) {
    p_ply ply_file = ply_open(filename.c_str(), nullptr, 0, nullptr);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}.",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;

    const char *element_name;
    long element_size;
    // Loop through ply elements and find "vertex".
    p_ply_element element = ply_get_next_element(ply_file, nullptr);
    while (element) {
        ply_get_element_info(element, &element_name, &element_size);
        if (std::string(element_name) == "vertex") {
            break;
        } else {
            element = ply_get_next_element(ply_file, element);
        }
    }

    // No element with name "vertex".
    if (!element) {
        utility::LogWarning("Read PLY failed: no vertex attribute.");
        return false;
    }

    p_ply_property attribute = ply_get_next_property(element, nullptr);

    while (attribute) {
        e_ply_type type;
        const char *name;
        ply_get_property_info(attribute, &name, &type, nullptr, nullptr);

        if (GetDtype(type) == core::Dtype::Undefined) {
            utility::LogWarning(
                    "Read PLY warning: skipping property \"{}\", unsupported "
                    "datatype \"{}\".",
                    name, GetDtypeString(type));
        } else {
            long size = 0;
            long id = static_cast<long>(state.id_to_attr_state_.size());
            DISPATCH_DTYPE_TO_TEMPLATE(GetDtype(type), [&]() {
                size = ply_set_read_cb(ply_file, element_name, name,
                                       ReadAttributeCallback<scalar_t>, &state,
                                       id);
            });
            if (size != element_size) {
                utility::LogError(
                        "Total size of property {} ({}) is not equal to "
                        "size of {} ({}).",
                        name, size, element_name, element_size);
            }

            auto attr_state = std::make_shared<PLYReaderState::AttrState>();
            attr_state->name_ = name;
            attr_state->data_ = core::Tensor({size}, GetDtype(type));
            attr_state->size_ = size;
            attr_state->current_size_ = 0;
            state.name_to_attr_state_.insert({name, attr_state});
            state.id_to_attr_state_.push_back(attr_state);
        }
        attribute = ply_get_next_property(element, attribute);
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(element_size);
    state.progress_bar_ = &reporter;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}.",
                            filename);
        ply_close(ply_file);
        return false;
    }

    pointcloud.Clear();

    // Add base attribtes.
    if (state.name_to_attr_state_.count("x") != 0 &&
        state.name_to_attr_state_.count("y") != 0 &&
        state.name_to_attr_state_.count("z") != 0) {
        core::TensorList points =
                ConcatColumns(state.name_to_attr_state_.at("x")->data_,
                              state.name_to_attr_state_.at("y")->data_,
                              state.name_to_attr_state_.at("z")->data_);
        state.name_to_attr_state_.erase("x");
        state.name_to_attr_state_.erase("y");
        state.name_to_attr_state_.erase("z");
        pointcloud.SetPoints(points);
    }
    if (state.name_to_attr_state_.count("nx") != 0 &&
        state.name_to_attr_state_.count("ny") != 0 &&
        state.name_to_attr_state_.count("nz") != 0) {
        core::TensorList normals =
                ConcatColumns(state.name_to_attr_state_.at("nx")->data_,
                              state.name_to_attr_state_.at("ny")->data_,
                              state.name_to_attr_state_.at("nz")->data_);
        state.name_to_attr_state_.erase("nx");
        state.name_to_attr_state_.erase("ny");
        state.name_to_attr_state_.erase("nz");
        pointcloud.SetPointNormals(normals);
    }
    if (state.name_to_attr_state_.count("red") != 0 &&
        state.name_to_attr_state_.count("green") != 0 &&
        state.name_to_attr_state_.count("blue") != 0) {
        core::TensorList colors =
                ConcatColumns(state.name_to_attr_state_.at("red")->data_,
                              state.name_to_attr_state_.at("green")->data_,
                              state.name_to_attr_state_.at("blue")->data_);
        state.name_to_attr_state_.erase("red");
        state.name_to_attr_state_.erase("green");
        state.name_to_attr_state_.erase("blue");
        pointcloud.SetPointColors(colors);
    }

    // Add rest of the attributes.
    for (auto const &it : state.name_to_attr_state_) {
        pointcloud.SetPointAttr(it.second->name_,
                                core::TensorList::FromTensor(it.second->data_));
    }
    ply_close(ply_file);
    reporter.Finish();

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d

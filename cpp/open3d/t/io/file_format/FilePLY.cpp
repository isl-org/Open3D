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

#include <rply.h>

#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

struct PLYReaderState {
    struct AttrState {
        std::string name_;
        void *data_ptr_;
        int stride_;
        int offset_;
        int64_t size_;
        int64_t current_size_;
    };
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

    T *data_ptr = static_cast<T *>(attr_state->data_ptr_);
    const int64_t index = attr_state->stride_ * attr_state->current_size_ +
                          attr_state->offset_;
    data_ptr[index] = static_cast<T>(ply_get_argument_value(argument));

    ++attr_state->current_size_;

    if (attr_state->offset_ == 0 && attr_state->current_size_ % 1000 == 0) {
        state_ptr->progress_bar_->Update(attr_state->current_size_);
    }
    return 1;
}

// Some of these datatypes are supported by Tensor but are added here just
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

    if (type == PLY_UINT8 || type == PLY_UCHAR) {
        return core::UInt8;
    } else if (type == PLY_UINT16) {
        return core::UInt16;
    } else if (type == PLY_INT32 || type == PLY_INT) {
        return core::Int32;
    } else if (type == PLY_FLOAT32 || type == PLY_FLOAT) {
        return core::Float32;
    } else if (type == PLY_FLOAT64 || type == PLY_DOUBLE) {
        return core::Float64;
    } else {
        return core::Undefined;
    }
}

static std::tuple<std::string, int, int> GetNameStrideOffsetForAttribute(
        const std::string &name) {
    // Positions attribute.
    if (name == "x") return std::make_tuple("positions", 3, 0);
    if (name == "y") return std::make_tuple("positions", 3, 1);
    if (name == "z") return std::make_tuple("positions", 3, 2);

    // Normals attribute.
    if (name == "nx") return std::make_tuple("normals", 3, 0);
    if (name == "ny") return std::make_tuple("normals", 3, 1);
    if (name == "nz") return std::make_tuple("normals", 3, 2);

    // Colors attribute.
    if (name == "red") return std::make_tuple("colors", 3, 0);
    if (name == "green") return std::make_tuple("colors", 3, 1);
    if (name == "blue") return std::make_tuple("colors", 3, 2);

    // Other attribute.
    return std::make_tuple(name, 1, 0);
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
    long element_size = 0;
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

    std::unordered_map<std::string, bool> primary_attr_init = {
            {"positions", false}, {"normals", false}, {"colors", false}};

    p_ply_property attribute = ply_get_next_property(element, nullptr);

    while (attribute) {
        e_ply_type type;
        const char *name;
        ply_get_property_info(attribute, &name, &type, nullptr, nullptr);

        if (GetDtype(type) == core::Undefined) {
            utility::LogWarning(
                    "Read PLY warning: skipping property \"{}\", unsupported "
                    "datatype \"{}\".",
                    name, GetDtypeString(type));
        } else {
            auto attr_state = std::make_shared<PLYReaderState::AttrState>();
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
            const std::string attr_name = std::string(name);

            std::tie(attr_state->name_, attr_state->stride_,
                     attr_state->offset_) =
                    GetNameStrideOffsetForAttribute(attr_name);

            if (primary_attr_init.count(attr_state->name_)) {
                if (primary_attr_init.at(attr_state->name_) == false) {
                    pointcloud.SetPointAttr(
                            attr_state->name_,
                            core::Tensor::Empty(
                                    {element_size, attr_state->stride_},
                                    GetDtype(type)));
                    primary_attr_init[attr_state->name_] = true;
                }
            } else {
                pointcloud.SetPointAttr(
                        attr_state->name_,
                        core::Tensor::Empty({element_size, attr_state->stride_},
                                            GetDtype(type)));
            }

            attr_state->data_ptr_ =
                    pointcloud.GetPointAttr(attr_state->name_).GetDataPtr();

            attr_state->size_ = element_size;
            attr_state->current_size_ = 0;
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

    ply_close(ply_file);
    reporter.Finish();

    return true;
}

static e_ply_type GetPlyType(const core::Dtype &dtype) {
    if (dtype == core::UInt8) {
        return PLY_UCHAR;
    } else if (dtype == core::UInt16) {
        return PLY_UINT16;
    } else if (dtype == core::Int32) {
        return PLY_INT;
    } else if (dtype == core::Float32) {
        return PLY_FLOAT;
    } else if (dtype == core::Float64) {
        return PLY_DOUBLE;
    } else {
        utility::LogError(
                "Data-type {} is not supported in WritePointCloudToPLY. "
                "Supported data-types include UInt8, UInt16, Int32, Float32 "
                "and Float64.",
                dtype.ToString());
    }
}

struct AttributePtr {
    AttributePtr(const core::Dtype &dtype,
                 const void *data_ptr,
                 const int &group_size)
        : dtype_(dtype), data_ptr_(data_ptr), group_size_(group_size) {}

    const core::Dtype dtype_;
    const void *data_ptr_;
    const int group_size_;
};

bool WritePointCloudToPLY(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const open3d::io::WritePointCloudOption &params) {
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write PLY failed: point cloud has 0 points.");
        return false;
    }

    geometry::TensorMap t_map(pointcloud.GetPointAttr().Contiguous());

    long num_points =
            static_cast<long>(pointcloud.GetPointPositions().GetLength());

    // Make sure all the attributes have same size.
    for (auto const &it : t_map) {
        if (it.first == "positions" || it.first == "normals" ||
            it.first == "colors") {
            if (it.second.GetLength() != num_points) {
                utility::LogWarning(
                        "Write PLY failed: Points ({}) and {} ({}) have "
                        "different lengths.",
                        num_points, it.first, it.second.GetLength());
                return false;
            }
        } else if (it.second.GetShape() != core::SizeVector({num_points, 1})) {
            utility::LogWarning(
                    "Write PLY failed. PointCloud contains {} attribute which "
                    "is not supported by PLY IO. Only points, normals, colors "
                    "and attributes with shape (num_points, 1) are supported. "
                    "Expected shape: {} but got {}.",
                    it.first, core::SizeVector({num_points, 1}).ToString(),
                    it.second.GetShape().ToString());
            return false;
        }
    }

    p_ply ply_file =
            ply_create(filename.c_str(),
                       bool(params.write_ascii) ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                       NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}.",
                            filename);
        return false;
    }

    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex", num_points);

    std::vector<AttributePtr> attribute_ptrs;
    attribute_ptrs.emplace_back(t_map["positions"].GetDtype(),
                                t_map["positions"].GetDataPtr(), 3);

    e_ply_type pointType = GetPlyType(t_map["positions"].GetDtype());
    ply_add_property(ply_file, "x", pointType, pointType, pointType);
    ply_add_property(ply_file, "y", pointType, pointType, pointType);
    ply_add_property(ply_file, "z", pointType, pointType, pointType);

    if (pointcloud.HasPointNormals()) {
        attribute_ptrs.emplace_back(t_map["normals"].GetDtype(),
                                    t_map["normals"].GetDataPtr(), 3);

        e_ply_type pointNormalsType = GetPlyType(t_map["normals"].GetDtype());
        ply_add_property(ply_file, "nx", pointNormalsType, pointNormalsType,
                         pointNormalsType);
        ply_add_property(ply_file, "ny", pointNormalsType, pointNormalsType,
                         pointNormalsType);
        ply_add_property(ply_file, "nz", pointNormalsType, pointNormalsType,
                         pointNormalsType);
    }

    if (pointcloud.HasPointColors()) {
        attribute_ptrs.emplace_back(t_map["colors"].GetDtype(),
                                    t_map["colors"].GetDataPtr(), 3);

        e_ply_type pointColorType = GetPlyType(t_map["colors"].GetDtype());
        ply_add_property(ply_file, "red", pointColorType, pointColorType,
                         pointColorType);
        ply_add_property(ply_file, "green", pointColorType, pointColorType,
                         pointColorType);
        ply_add_property(ply_file, "blue", pointColorType, pointColorType,
                         pointColorType);
    }

    e_ply_type attributeType;
    for (auto const &it : t_map) {
        if (it.first != "positions" && it.first != "colors" &&
            it.first != "normals") {
            attribute_ptrs.emplace_back(it.second.GetDtype(),
                                        it.second.GetDataPtr(), 1);

            attributeType = GetPlyType(it.second.GetDtype());
            ply_add_property(ply_file, it.first.c_str(), attributeType,
                             attributeType, attributeType);
        }
    }

    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.");
        ply_close(ply_file);
        return false;
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_points);

    for (int64_t i = 0; i < num_points; i++) {
        for (auto it : attribute_ptrs) {
            DISPATCH_DTYPE_TO_TEMPLATE(it.dtype_, [&]() {
                const scalar_t *data_ptr =
                        static_cast<const scalar_t *>(it.data_ptr_);
                for (int idx_offset = it.group_size_ * i;
                     idx_offset < it.group_size_ * (i + 1); ++idx_offset) {
                    ply_write(ply_file, data_ptr[idx_offset]);
                }
            });
        }

        if (i % 1000 == 0) {
            reporter.Update(i);
        }
    }

    reporter.Finish();
    ply_close(ply_file);
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d

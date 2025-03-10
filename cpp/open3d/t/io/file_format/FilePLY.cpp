// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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

#define POSITIONS_ATTR_NAME "positions"
#define NORMALS_ATTR_NAME "normals"
#define COLORS_ATTR_NAME "colors"
#define OPACITY_ATTR_NAME "opacity"  // The opacity
#define ROTATION_ATTR_NAME "rot"     // The Gaussian rotation as a quaternion.
#define SCALE_ATTR_NAME "scale"      // The Gaussian scale.
#define F_DC_ATTR_NAME "f_dc"        // DC color components.
#define F_REST_ATTR_NAME "f_rest"    // Spherical Harmonics Coefficients (SHC).

#define POSITIONS_ATTR_MAX_DIM 3
#define NORMALS_ATTR_MAX_DIM 3
#define COLORS_ATTR_MAX_DIM 3
#define OPACITY_ATTR_MAX_DIM 1   // Scalar opacity.
#define ROTATION_ATTR_MAX_DIM 4  // Rotation channels.
#define SCALE_ATTR_MAX_DIM 3     // Channels for x,y,z.
#define F_DC_ATTR_MAX_DIM 3      // Channels for R,G,B.
#define F_REST_ATTR_MAX_DIM 1  // Non-DC color. Each property read as a scalar.

#define ROTATION_SUFFIX_INDEX 4
#define SCALE_SUFFIX_INDEX 6
#define F_DC_SUFFIX_INDEX 5
#define F_REST_SUFFIX_INDEX 7

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
    if (name == "x")
        return std::make_tuple(POSITIONS_ATTR_NAME, POSITIONS_ATTR_MAX_DIM, 0);
    if (name == "y")
        return std::make_tuple(POSITIONS_ATTR_NAME, POSITIONS_ATTR_MAX_DIM, 1);
    if (name == "z")
        return std::make_tuple(POSITIONS_ATTR_NAME, POSITIONS_ATTR_MAX_DIM, 2);

    // Normals attribute.
    if (name == "nx")
        return std::make_tuple(NORMALS_ATTR_NAME, NORMALS_ATTR_MAX_DIM, 0);
    if (name == "ny")
        return std::make_tuple(NORMALS_ATTR_NAME, NORMALS_ATTR_MAX_DIM, 1);
    if (name == "nz")
        return std::make_tuple(NORMALS_ATTR_NAME, NORMALS_ATTR_MAX_DIM, 2);

    // Colors attribute.
    if (name == "red")
        return std::make_tuple(COLORS_ATTR_NAME, COLORS_ATTR_MAX_DIM, 0);
    if (name == "green")
        return std::make_tuple(COLORS_ATTR_NAME, COLORS_ATTR_MAX_DIM, 1);
    if (name == "blue")
        return std::make_tuple(COLORS_ATTR_NAME, COLORS_ATTR_MAX_DIM, 2);

    // Extra attributes for 3DGS.
    // Property names with prefixes are mapped to grouped attributes.

    // f_dc attribute.
    if (name.rfind(std::string(F_DC_ATTR_NAME) + "_", 0) == 0) {
        int offset = std::stoi(name.substr(F_DC_SUFFIX_INDEX));
        return std::make_tuple(F_DC_ATTR_NAME, F_DC_ATTR_MAX_DIM, offset);
    }
    // f_rest attribute.
    if (name.rfind(std::string(F_REST_ATTR_NAME) + "_", 0) == 0) {
        int offset = std::stoi(name.substr(F_REST_SUFFIX_INDEX));
        return std::make_tuple(F_REST_ATTR_NAME, F_REST_ATTR_MAX_DIM, offset);
    }
    // scale attribute.
    if (name.rfind(std::string(SCALE_ATTR_NAME) + "_", 0) == 0) {
        int offset = std::stoi(name.substr(SCALE_SUFFIX_INDEX));
        return std::make_tuple(SCALE_ATTR_NAME, SCALE_ATTR_MAX_DIM, offset);
    }
    // rot attribute.
    if (name.rfind(std::string(ROTATION_ATTR_NAME) + "_", 0) == 0) {
        int offset = std::stoi(name.substr(ROTATION_SUFFIX_INDEX));
        return std::make_tuple(ROTATION_ATTR_NAME, ROTATION_ATTR_MAX_DIM,
                               offset);
    }
    // Other attribute.
    return std::make_tuple(name, 1, 0);
}

bool Is3DGSPointCloud(const geometry::PointCloud &pointcloud) {
    return (pointcloud.HasPointAttr(OPACITY_ATTR_NAME) &&
            pointcloud.HasPointAttr(ROTATION_ATTR_NAME) &&
            pointcloud.HasPointAttr(SCALE_ATTR_NAME) &&
            pointcloud.HasPointAttr(F_DC_ATTR_NAME));
}

bool CheckMandatory3DGSProperty(const geometry::PointCloud &pointcloud,
                                const std::string &property,
                                const core::SizeVector &dims) {
    core::AssertTensorShape(pointcloud.GetPointAttr(property), dims);
    return true;
}

bool Validate3DGSPointCloudProperties(const geometry::PointCloud &pointcloud) {
    int64_t num_points = pointcloud.GetPointPositions().GetLength();

    // Assert attribute shapes.
    bool valid_positions =
            CheckMandatory3DGSProperty(pointcloud, POSITIONS_ATTR_NAME,
                                       {num_points, POSITIONS_ATTR_MAX_DIM});
    bool valid_normals = CheckMandatory3DGSProperty(
            pointcloud, NORMALS_ATTR_NAME, {num_points, NORMALS_ATTR_MAX_DIM});
    bool valid_opacity = CheckMandatory3DGSProperty(
            pointcloud, OPACITY_ATTR_NAME, {num_points, OPACITY_ATTR_MAX_DIM});
    bool valid_rot =
            CheckMandatory3DGSProperty(pointcloud, ROTATION_ATTR_NAME,
                                       {num_points, ROTATION_ATTR_MAX_DIM});
    bool valid_scale = CheckMandatory3DGSProperty(
            pointcloud, SCALE_ATTR_NAME, {num_points, SCALE_ATTR_MAX_DIM});
    bool valid_f_dc = CheckMandatory3DGSProperty(
            pointcloud, F_DC_ATTR_NAME, {num_points, F_DC_ATTR_MAX_DIM});

    return (valid_positions && valid_normals && valid_opacity && valid_rot &&
            valid_scale && valid_f_dc);
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
        ply_close(ply_file);
        return false;
    }

    std::unordered_map<std::string, bool> primary_attr_init = {
            {POSITIONS_ATTR_NAME, false}, {NORMALS_ATTR_NAME, false},
            {COLORS_ATTR_NAME, false},    {OPACITY_ATTR_NAME, false},
            {ROTATION_ATTR_NAME, false},  {SCALE_ATTR_NAME, false},
            {F_DC_ATTR_NAME, false},      {F_REST_ATTR_NAME, false},
    };

    int f_rest_count = 0;
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

            std::string target;
            int stride, offset;

            std::tie(target, stride, offset) =
                    GetNameStrideOffsetForAttribute(attr_name);

            if (target == F_REST_ATTR_NAME) {
                stride = 1;
                f_rest_count++;
            }

            attr_state->name_ = target;
            attr_state->stride_ = stride;
            attr_state->offset_ = offset;

            if (primary_attr_init.count(target)) {
                if (primary_attr_init.at(target) == false) {
                    pointcloud.SetPointAttr(
                            target, core::Tensor::Empty({element_size, stride},
                                                        GetDtype(type)));
                    primary_attr_init[target] = true;
                }
            } else {
                pointcloud.SetPointAttr(
                        target, core::Tensor::Empty({element_size, stride},
                                                    GetDtype(type)));
            }

            attr_state->data_ptr_ =
                    pointcloud.GetPointAttr(target).GetDataPtr();

            attr_state->size_ = element_size;
            attr_state->current_size_ = 0;
            state.id_to_attr_state_.push_back(attr_state);
        }

        attribute = ply_get_next_property(element, attribute);
    }

    if (f_rest_count > 0) {
        core::Tensor new_f_rest =
                core::Tensor::Empty({element_size, f_rest_count}, core::Float32,
                                    pointcloud.GetDevice());
        pointcloud.SetPointAttr(F_REST_ATTR_NAME, new_f_rest);
        for (auto &attr_state : state.id_to_attr_state_) {
            if (attr_state->name_ == F_REST_ATTR_NAME) {
                attr_state->data_ptr_ =
                        pointcloud.GetPointAttr(F_REST_ATTR_NAME).GetDataPtr();
                attr_state->stride_ = 1;
            }
        }
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

    if (Is3DGSPointCloud(pointcloud)) {
        utility::LogInfo("PLY file identified as 3DGS format.");
        if (!Validate3DGSPointCloudProperties(pointcloud)) {
            return false;
        }
    }

    if (pointcloud.HasPointAttr(F_REST_ATTR_NAME)) {
        core::Tensor f_rest_tensor = pointcloud.GetPointAttr(F_REST_ATTR_NAME);
        auto shape = f_rest_tensor.GetShape();  // shape = {N, f_rest_count}
        utility::LogInfo("f_rest tensor shape before: {}", shape.ToString());
        int num_f_rest_channels = shape[1];
        utility::LogInfo("Computed num_f_rest_channels: {}",
                         num_f_rest_channels);
        if (num_f_rest_channels % 3 == 0) {
            int Nc = num_f_rest_channels / 3;
            pointcloud.SetPointAttr(
                    F_REST_ATTR_NAME,
                    f_rest_tensor.Reshape({shape[0], Nc, 3}).Clone());
        } else {
            utility::LogWarning(
                    "f_rest attribute has {} columns, which is not divisible "
                    "by 3.",
                    num_f_rest_channels);
        }
    }
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

    if (Is3DGSPointCloud(pointcloud)) {
        utility::LogInfo("PLY point cloud identified as 3DGS format.");
        if (!Validate3DGSPointCloudProperties(pointcloud)) {
            return false;
        }
    }

    geometry::TensorMap t_map(pointcloud.GetPointAttr().Contiguous());

    long num_points =
            static_cast<long>(pointcloud.GetPointPositions().GetLength());

    // Verify that standard attributes have length equal to num_points.
    // Extra attributes must have at least 2 dimensions: (num_points, channels).
    for (auto const &it : t_map) {
        if (it.first == POSITIONS_ATTR_NAME || it.first == NORMALS_ATTR_NAME ||
            it.first == COLORS_ATTR_NAME) {
            if (it.second.GetLength() != num_points) {
                utility::LogWarning(
                        "Write PLY failed: Points ({}) and {} ({}) have "
                        "different lengths.",
                        num_points, it.first, it.second.GetLength());
                return false;
            }
        } else {
            auto shape = it.second.GetShape();
            // Only tensors with shape (num_points, channels) are supported.
            if (shape.size() < 2 || shape[0] != num_points) {
                utility::LogWarning(
                        "Write PLY failed. PointCloud contains {} attribute "
                        "which "
                        "is not supported by PLY IO. Only points, normals, "
                        "colors "
                        "and attributes with shape (num_points, 1) are "
                        "supported. "
                        "Expected shape: {} but got {}.",
                        it.first, core::SizeVector({num_points, 1}).ToString(),
                        it.second.GetShape().ToString());
                return false;
            }
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
    attribute_ptrs.emplace_back(t_map[POSITIONS_ATTR_NAME].GetDtype(),
                                t_map[POSITIONS_ATTR_NAME].GetDataPtr(),
                                POSITIONS_ATTR_MAX_DIM);

    e_ply_type pointType = GetPlyType(t_map[POSITIONS_ATTR_NAME].GetDtype());
    ply_add_property(ply_file, "x", pointType, pointType, pointType);
    ply_add_property(ply_file, "y", pointType, pointType, pointType);
    ply_add_property(ply_file, "z", pointType, pointType, pointType);

    if (pointcloud.HasPointNormals()) {
        attribute_ptrs.emplace_back(t_map[NORMALS_ATTR_NAME].GetDtype(),
                                    t_map[NORMALS_ATTR_NAME].GetDataPtr(),
                                    NORMALS_ATTR_MAX_DIM);

        e_ply_type pointNormalsType =
                GetPlyType(t_map[NORMALS_ATTR_NAME].GetDtype());
        ply_add_property(ply_file, "nx", pointNormalsType, pointNormalsType,
                         pointNormalsType);
        ply_add_property(ply_file, "ny", pointNormalsType, pointNormalsType,
                         pointNormalsType);
        ply_add_property(ply_file, "nz", pointNormalsType, pointNormalsType,
                         pointNormalsType);
    }

    if (pointcloud.HasPointColors()) {
        attribute_ptrs.emplace_back(t_map[COLORS_ATTR_NAME].GetDtype(),
                                    t_map[COLORS_ATTR_NAME].GetDataPtr(),
                                    COLORS_ATTR_MAX_DIM);

        e_ply_type pointColorType =
                GetPlyType(t_map[COLORS_ATTR_NAME].GetDtype());
        ply_add_property(ply_file, "red", pointColorType, pointColorType,
                         pointColorType);
        ply_add_property(ply_file, "green", pointColorType, pointColorType,
                         pointColorType);
        ply_add_property(ply_file, "blue", pointColorType, pointColorType,
                         pointColorType);
    }

    // Process extra attributes.
    // Extra attributes are expected to be tensors with shape (num_points,
    // channels) or (num_points, C, D). For multi-channel attributes, the
    // channels are flattened, and each channel is written as a separate
    // property (e.g., "f_rest_0", "f_rest_1", ...).
    e_ply_type attributeType;
    for (auto const &it : t_map) {
        if (it.first == POSITIONS_ATTR_NAME || it.first == COLORS_ATTR_NAME ||
            it.first == NORMALS_ATTR_NAME)
            continue;
        auto shape = it.second.GetShape();
        int group_size = 1;
        if (shape.size() == 2) {
            group_size = shape[1];
        } else if (shape.size() >= 3) {
            group_size = shape[1] * shape[2];
        }
        if (group_size == 1) {
            attribute_ptrs.emplace_back(it.second.GetDtype(),
                                        it.second.GetDataPtr(), 1);
            attributeType = GetPlyType(it.second.GetDtype());
            ply_add_property(ply_file, it.first.c_str(), attributeType,
                             attributeType, attributeType);
        } else {
            for (int ch = 0; ch < group_size; ch++) {
                std::string prop_name = it.first + "_" + std::to_string(ch);
                attributeType = GetPlyType(it.second.GetDtype());
                ply_add_property(ply_file, prop_name.c_str(), attributeType,
                                 attributeType, attributeType);
            }
            attribute_ptrs.emplace_back(it.second.GetDtype(),
                                        it.second.GetDataPtr(), group_size);
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

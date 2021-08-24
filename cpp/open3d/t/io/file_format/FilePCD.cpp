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

#include <liblzf/lzf.h>

#include <cstdint>
#include <cstdio>
#include <sstream>

#include "open3d/core/Dtype.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

// References for PCD file IO
// http://pointclouds.org/documentation/tutorials/pcd_file_format.html
// https://github.com/PointCloudLibrary/pcl/blob/master/io/src/pcd_io.cpp
// https://www.mathworks.com/matlabcentral/fileexchange/40382-matlab-to-point-cloud-library

namespace open3d {
namespace t {
namespace io {

namespace {
using namespace io;

enum PCDDataType {
    PCD_DATA_ASCII = 0,
    PCD_DATA_BINARY = 1,
    PCD_DATA_BINARY_COMPRESSED = 2
};

struct PCLPointField {
public:
    std::string name;
    int size;
    char type;
    int count;
    // helper variable
    int count_offset;
    int offset;
};

struct PCDHeader {
public:
    std::string version;
    std::vector<PCLPointField> fields;
    int width;
    int height;
    int points;
    PCDDataType datatype;
    std::string viewpoint;

    // helper variables
    int elementnum;
    int pointsize;
    std::unordered_map<std::string, bool> has_attr;
    std::unordered_map<std::string, core::Dtype> attr_dtype;
};

core::Dtype GetDtypeFromPCDHeaderField(char type, int size) {
    if (type == 'I') {
        if (size == 1) {
            return core::Dtype::Int8;
        } else if (size == 2) {
            return core::Dtype::Int16;
        } else if (size == 4) {
            return core::Dtype::Int32;
        } else if (size == 8) {
            return core::Dtype::Int64;
        } else {
            utility::LogError("Unsupported data type.");
        }
    } else if (type == 'U') {
        if (size == 1) {
            return core::Dtype::UInt8;
        } else if (size == 2) {
            return core::Dtype::UInt16;
        } else if (size == 4) {
            return core::Dtype::UInt32;
        } else if (size == 8) {
            return core::Dtype::UInt64;
        } else {
            utility::LogError("Unsupported data type.");
        }
    } else if (type == 'F') {
        if (size == 4) {
            return core::Dtype::Float32;
        } else if (size == 8) {
            return core::Dtype::Float64;
        } else {
            utility::LogError("Unsupported data type.");
        }
    } else {
        utility::LogError("Unsupported data type.");
    }
}

bool CheckHeader(PCDHeader &header) {
    if (header.points <= 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no data.");
        return false;
    }
    if (header.fields.size() == 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no fields.");
        return false;
    }
    header.has_attr["positions"] = false;
    header.has_attr["normals"] = false;
    header.has_attr["colors"] = false;

    bool has_x = false;
    bool has_y = false;
    bool has_z = false;
    bool has_normal_x = false;
    bool has_normal_y = false;
    bool has_normal_z = false;
    bool has_rgb = false;
    bool has_rgba = false;

    core::Dtype dtype_x;
    core::Dtype dtype_y;
    core::Dtype dtype_z;
    core::Dtype dtype_normals_x;
    core::Dtype dtype_normals_y;
    core::Dtype dtype_normals_z;
    core::Dtype dtype_colors;
    core::Dtype dtype_rgba;

    for (const auto &field : header.fields) {
        if (field.name == "x") {
            has_x = true;
            dtype_x = GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "y") {
            has_y = true;
            dtype_y = GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "z") {
            has_z = true;
            dtype_z = GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "normal_x") {
            has_normal_x = true;
            dtype_normals_x =
                    GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "normal_y") {
            has_normal_y = true;
            dtype_normals_y =
                    GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "normal_z") {
            has_normal_z = true;
            dtype_normals_z =
                    GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "rgb") {
            has_rgb = true;
            dtype_colors = GetDtypeFromPCDHeaderField(field.type, field.size);
        } else if (field.name == "rgba") {
            has_rgba = true;
            dtype_colors = GetDtypeFromPCDHeaderField(field.type, field.size);
        } else {
            // Support for custom attribute field with shape
            // {num_points, 1}.
            header.has_attr[field.name] = true;
            header.attr_dtype[field.name] =
                    GetDtypeFromPCDHeaderField(field.type, field.size);
        }
    }

    if (has_x && has_y && has_z) {
        if ((dtype_x == dtype_y) && (dtype_x == dtype_z)) {
            header.has_attr["positions"] = true;
            header.attr_dtype["positions"] = dtype_x;
        } else {
            utility::LogWarning(
                    "[CheckHeader] Dtype for positions data are not same.");
            return false;
        }
    } else {
        utility::LogWarning(
                "[CheckHeader] Fields for positions data are not complete.");
        return false;
    }

    if (has_normal_x && has_normal_y && has_normal_z) {
        if ((dtype_normals_x == dtype_normals_y) &&
            (dtype_normals_x == dtype_normals_z)) {
            header.has_attr["normals"] = true;
            header.attr_dtype["normals"] = dtype_normals_x;
        } else {
            utility::LogWarning(
                    "[CheckHeader] Dtype for normals data are not same.");
            return false;
        }
    }

    if (has_rgb || has_rgba) {
        header.has_attr["colors"] = true;
        header.attr_dtype["colors"] = dtype_colors;
    }

    return true;
}

bool ReadPCDHeader(FILE *file, PCDHeader &header) {
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    size_t specified_channel_count = 0;

    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::string line(line_buffer);
        if (line == "") {
            continue;
        }
        std::vector<std::string> st = utility::SplitString(line, "\t\r\n ");
        std::stringstream sstream(line);
        sstream.imbue(std::locale::classic());
        std::string line_type;
        sstream >> line_type;
        if (line_type.substr(0, 1) == "#") {
        } else if (line_type.substr(0, 7) == "VERSION") {
            if (st.size() >= 2) {
                header.version = st[1];
            }
        } else if (line_type.substr(0, 6) == "FIELDS" ||
                   line_type.substr(0, 7) == "COLUMNS") {
            specified_channel_count = st.size() - 1;
            if (specified_channel_count == 0) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            header.fields.resize(specified_channel_count);
            int count_offset = 0, offset = 0;
            for (size_t i = 0; i < specified_channel_count;
                 ++i, count_offset += 1, offset += 4) {
                header.fields[i].name = st[i + 1];
                header.fields[i].size = 4;
                header.fields[i].type = 'F';
                header.fields[i].count = 1;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "SIZE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            int offset = 0, col_type = 0;
            for (size_t i = 0; i < specified_channel_count;
                 ++i, offset += col_type) {
                sstream >> col_type;
                header.fields[i].size = col_type;
                header.fields[i].offset = offset;
            }
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "TYPE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            for (size_t i = 0; i < specified_channel_count; ++i) {
                header.fields[i].type = st[i + 1].c_str()[0];
            }
        } else if (line_type.substr(0, 5) == "COUNT") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            int count_offset = 0, offset = 0, col_count = 0;
            for (size_t i = 0; i < specified_channel_count; ++i) {
                sstream >> col_count;
                header.fields[i].count = col_count;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
                count_offset += col_count;
                offset += col_count * header.fields[i].size;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 5) == "WIDTH") {
            sstream >> header.width;
        } else if (line_type.substr(0, 6) == "HEIGHT") {
            sstream >> header.height;
            header.points = header.width * header.height;
        } else if (line_type.substr(0, 9) == "VIEWPOINT") {
            if (st.size() >= 2) {
                header.viewpoint = st[1];
            }
        } else if (line_type.substr(0, 6) == "POINTS") {
            sstream >> header.points;
        } else if (line_type.substr(0, 4) == "DATA") {
            header.datatype = PCD_DATA_ASCII;
            if (st.size() >= 2) {
                if (st[1].substr(0, 17) == "binary_compressed") {
                    header.datatype = PCD_DATA_BINARY_COMPRESSED;
                } else if (st[1].substr(0, 6) == "binary") {
                    header.datatype = PCD_DATA_BINARY;
                }
            }
            break;
        }
    }
    if (!CheckHeader(header)) {
        return false;
    }
    return true;
}

struct ReadAttributePtr {
    ReadAttributePtr(void *data_ptr = nullptr,
                     const int row_idx = 0,
                     const int row_length = 0,
                     const int size = 0)
        : data_ptr_(data_ptr),
          row_idx_(row_idx),
          row_length_(row_length),
          size_(size) {}

    void *data_ptr_;
    const int row_idx_;
    const int row_length_;
    const int size_;
};

inline void ReadASCIIPCDColorsFromField(ReadAttributePtr &attr,
                                        const PCLPointField &field,
                                        const char *data_ptr,
                                        const int index) {
    uint8_t *attr_data_ptr =
            static_cast<uint8_t *>(attr.data_ptr_) + index * attr.row_length_;

    if (field.size == 4) {
        std::uint8_t data[4] = {0};
        char *end;
        if (field.type == 'I') {
            std::int32_t value = std::strtol(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (field.type == 'U') {
            std::uint32_t value = std::strtoul(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (field.type == 'F') {
            float value = std::strtof(data_ptr, &end);
            memcpy(data, &value, 4);
        }

        // color data is packed in BGR order.
        attr_data_ptr[0] = data[2];
        attr_data_ptr[1] = data[1];
        attr_data_ptr[2] = data[0];
    } else {
        attr_data_ptr[0] = 0;
        attr_data_ptr[1] = 0;
        attr_data_ptr[2] = 0;
    }
}

inline void ReadBinaryPCDColorsFromField(ReadAttributePtr &attr,
                                         const PCLPointField &field,
                                         const void *data_ptr,
                                         const int index) {
    uint8_t *attr_data_ptr =
            static_cast<uint8_t *>(attr.data_ptr_) + index * attr.row_length_;

    if (field.size == 4) {
        std::uint8_t data[4] = {0};
        memcpy(data, data_ptr, 4);

        // color data is packed in BGR order.
        attr_data_ptr[0] = data[2];
        attr_data_ptr[1] = data[1];
        attr_data_ptr[2] = data[0];
    } else {
        attr_data_ptr[0] = 0;
        attr_data_ptr[1] = 0;
        attr_data_ptr[2] = 0;
    }
}

inline void ReadPCDElementsFromField(ReadAttributePtr &attr,
                                     const PCLPointField &field,
                                     const void *data_ptr,
                                     const int index) {
    DISPATCH_DTYPE_TO_TEMPLATE(
            GetDtypeFromPCDHeaderField(field.type, field.size), [&] {
                scalar_t *attr_data_ptr =
                        static_cast<scalar_t *>(attr.data_ptr_) +
                        index * attr.row_length_ + attr.row_idx_;

                memcpy(attr_data_ptr, data_ptr, sizeof(scalar_t));
            });
}

bool ReadPCDData(FILE *file,
                 PCDHeader &header,
                 t::geometry::PointCloud &pointcloud,
                 const ReadPointCloudOption &params) {
    // The header should have been checked
    pointcloud.Clear();

    std::unordered_map<std::string, ReadAttributePtr> map_field_to_attr_ptr;

    if (header.has_attr["positions"]) {
        pointcloud.SetPointPositions(core::Tensor::Empty(
                {header.points, 3}, header.attr_dtype["positions"]));

        void *data_ptr = pointcloud.GetPointPositions().GetDataPtr();
        ReadAttributePtr position_x(data_ptr, 0, 3, header.points);
        ReadAttributePtr position_y(data_ptr, 1, 3, header.points);
        ReadAttributePtr position_z(data_ptr, 2, 3, header.points);

        map_field_to_attr_ptr.emplace(std::string("x"), position_x);
        map_field_to_attr_ptr.emplace(std::string("y"), position_y);
        map_field_to_attr_ptr.emplace(std::string("z"), position_z);
    } else {
        utility::LogWarning(
                "[ReadPCDData] Fields for point data are not complete.");
        return false;
    }
    if (header.has_attr["normals"]) {
        pointcloud.SetPointNormals(core::Tensor::Empty(
                {header.points, 3}, header.attr_dtype["normals"]));

        void *data_ptr = pointcloud.GetPointNormals().GetDataPtr();
        ReadAttributePtr normal_x(data_ptr, 0, 3, header.points);
        ReadAttributePtr normal_y(data_ptr, 1, 3, header.points);
        ReadAttributePtr normal_z(data_ptr, 2, 3, header.points);

        map_field_to_attr_ptr.emplace(std::string("normal_x"), normal_x);
        map_field_to_attr_ptr.emplace(std::string("normal_y"), normal_y);
        map_field_to_attr_ptr.emplace(std::string("normal_z"), normal_z);
    }
    if (header.has_attr["colors"]) {
        // Colors stored in a PCD file is ALWAYS in UInt8 format.
        // However it is stored as a single packed floating value.
        pointcloud.SetPointColors(
                core::Tensor::Empty({header.points, 3}, core::UInt8));

        void *data_ptr = pointcloud.GetPointColors().GetDataPtr();
        ReadAttributePtr colors(data_ptr, 0, 3, header.points);

        map_field_to_attr_ptr.emplace(std::string("colors"), colors);
    }

    // PCLPointField
    for (const auto &field : header.fields) {
        if (header.has_attr[field.name] && field.name != "x" &&
            field.name != "y" && field.name != "z" &&
            field.name != "normal_x" && field.name != "normal_y" &&
            field.name != "normal_z" && field.name != "rgb" &&
            field.name != "rgba") {
            pointcloud.SetPointAttr(
                    field.name,
                    core::Tensor::Empty({header.points, 1},
                                        header.attr_dtype[field.name]));

            void *data_ptr = pointcloud.GetPointAttr(field.name).GetDataPtr();
            ReadAttributePtr attr(data_ptr, 0, 1, header.points);

            map_field_to_attr_ptr.emplace(field.name, attr);
        }
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(header.points);

    if (header.datatype == PCD_DATA_ASCII) {
        char line_buffer[DEFAULT_IO_BUFFER_SIZE];
        int idx = 0;
        while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file) &&
               idx < header.points) {
            std::string line(line_buffer);
            std::vector<std::string> strs =
                    utility::SplitString(line, "\t\r\n ");
            if ((int)strs.size() < header.elementnum) {
                continue;
            }
            for (size_t i = 0; i < header.fields.size(); ++i) {
                const auto &field = header.fields[i];
                if (field.name == "rgb" || field.name == "rgba") {
                    ReadASCIIPCDColorsFromField(
                            map_field_to_attr_ptr["colors"], field,
                            strs[field.count_offset].c_str(), idx);
                } else {
                    ReadPCDElementsFromField(
                            map_field_to_attr_ptr[field.name], field,
                            strs[field.count_offset].c_str(), idx);
                }
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::unique_ptr<char[]> buffer(new char[header.pointsize]);
        for (int i = 0; i < header.points; ++i) {
            if (fread(buffer.get(), header.pointsize, 1, file) != 1) {
                utility::LogWarning(
                        "[ReadPCDData] Failed to read data record.");
                pointcloud.Clear();
                return false;
            }
            for (const auto &field : header.fields) {
                if (field.name == "rgb" || field.name == "rgba") {
                    ReadBinaryPCDColorsFromField(
                            map_field_to_attr_ptr["colors"], field,
                            buffer.get() + field.offset, i);
                } else {
                    ReadPCDElementsFromField(map_field_to_attr_ptr[field.name],
                                             field, buffer.get() + field.offset,
                                             i);
                }
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        double reporter_total = 100.0;
        reporter.SetTotal(int(reporter_total));
        reporter.Update(int(reporter_total * 0.01));
        std::uint32_t compressed_size;
        std::uint32_t uncompressed_size;
        if (fread(&compressed_size, sizeof(compressed_size), 1, file) != 1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.Clear();
            return false;
        }
        if (fread(&uncompressed_size, sizeof(uncompressed_size), 1, file) !=
            1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.Clear();
            return false;
        }
        utility::LogDebug(
                "PCD data with {:d} compressed size, and {:d} uncompressed "
                "size.",
                compressed_size, uncompressed_size);
        std::unique_ptr<char[]> buffer_compressed(new char[compressed_size]);
        reporter.Update(int(reporter_total * .1));
        if (fread(buffer_compressed.get(), 1, compressed_size, file) !=
            compressed_size) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.Clear();
            return false;
        }
        std::unique_ptr<char[]> buffer(new char[uncompressed_size]);
        reporter.Update(int(reporter_total * .2));
        if (lzf_decompress(buffer_compressed.get(),
                           (unsigned int)compressed_size, buffer.get(),
                           (unsigned int)uncompressed_size) !=
            uncompressed_size) {
            utility::LogWarning("[ReadPCDData] Uncompression failed.");
            pointcloud.Clear();
            return false;
        }
        for (const auto &field : header.fields) {
            const char *base_ptr = buffer.get() + field.offset * header.points;
            double progress =
                    double(base_ptr - buffer.get()) / uncompressed_size;
            reporter.Update(int(reporter_total * (progress + .2)));
            if (field.name == "rgb" || field.name == "rgba") {
                for (int i = 0; i < header.points; ++i) {
                    ReadBinaryPCDColorsFromField(
                            map_field_to_attr_ptr["colors"], field,
                            base_ptr + i * field.size * field.count, i);
                }
            } else {
                for (int i = 0; i < header.points; ++i) {
                    ReadPCDElementsFromField(
                            map_field_to_attr_ptr[field.name], field,
                            base_ptr + i * field.size * field.count, i);
                }
            }
        }
    }
    reporter.Finish();
    return true;
}

// Write functions.

void SetPCDHeaderFieldTypeAndSizeFromDtype(const core::Dtype &dtype,
                                           PCLPointField &field) {
    if (dtype == core::Int8) {
        field.type = 'I';
        field.size = 1;
    } else if (dtype == core::Int16) {
        field.type = 'I';
        field.size = 2;
    } else if (dtype == core::Int32) {
        field.type = 'I';
        field.size = 4;
    } else if (dtype == core::Int64) {
        field.type = 'I';
        field.size = 8;
    } else if (dtype == core::UInt8) {
        field.type = 'U';
        field.size = 1;
    } else if (dtype == core::UInt16) {
        field.type = 'U';
        field.size = 2;
    } else if (dtype == core::UInt32) {
        field.type = 'U';
        field.size = 4;
    } else if (dtype == core::UInt64) {
        field.type = 'U';
        field.size = 8;
    } else if (dtype == core::Float32) {
        field.type = 'F';
        field.size = 4;
    } else if (dtype == core::Float64) {
        field.type = 'F';
        field.size = 8;
    } else {
        utility::LogError("Unsupported data type.");
    }
}

bool GenerateHeader(const t::geometry::PointCloud &pointcloud,
                    const bool write_ascii,
                    const bool compressed,
                    PCDHeader &header) {
    if (!pointcloud.HasPointPositions()) {
        return false;
    }

    header.version = "0.7";
    header.width = static_cast<int>(pointcloud.GetPointPositions().GetLength());
    header.height = 1;
    header.points = header.width;
    header.fields.clear();

    PCLPointField field_x, field_y, field_z;
    field_x.name = "x";
    field_x.count = 1;
    SetPCDHeaderFieldTypeAndSizeFromDtype(
            pointcloud.GetPointPositions().GetDtype(), field_x);
    header.fields.push_back(field_x);

    field_y.name = "y";
    field_y.count = 1;
    field_y.type = field_x.type;
    field_y.size = field_x.size;
    header.fields.push_back(field_y);

    field_z.name = "z";
    field_z.count = 1;
    field_z.type = field_x.type;
    field_z.size = field_x.size;
    header.fields.push_back(field_z);

    header.elementnum = 3 * field_x.count;
    header.pointsize = 3 * field_x.size;

    if (pointcloud.HasPointNormals()) {
        PCLPointField field_normal_x, field_normal_y, field_normal_z;

        field_normal_x.name = "normal_x";
        field_normal_x.count = 1;
        SetPCDHeaderFieldTypeAndSizeFromDtype(
                pointcloud.GetPointPositions().GetDtype(), field_normal_x);
        header.fields.push_back(field_normal_x);

        field_normal_y.name = "normal_y";
        field_normal_y.count = 1;
        field_normal_y.type = field_normal_x.type;
        field_normal_y.size = field_normal_x.size;
        header.fields.push_back(field_normal_y);

        field_normal_z.name = "normal_z";
        field_normal_z.count = 1;
        field_normal_z.type = field_normal_x.type;
        field_normal_z.size = field_normal_x.size;
        header.fields.push_back(field_normal_z);

        header.elementnum += 3 * field_normal_x.count;
        header.pointsize += 3 * field_normal_x.size;
    }
    if (pointcloud.HasPointColors()) {
        PCLPointField field_colors;
        field_colors.name = "rgb";
        field_colors.count = 1;
        field_colors.type = 'F';
        field_colors.size = 4;
        header.fields.push_back(field_colors);

        header.elementnum++;
        header.pointsize += 4;
    }

    // Custom attribute support of shape {num_points, 1}.
    for (auto &kv : pointcloud.GetPointAttr()) {
        if (kv.first != "positions" && kv.first != "normals" &&
            kv.first != "colors") {
            if (kv.second.GetShape() ==
                core::SizeVector(
                        {pointcloud.GetPointPositions().GetLength(), 1})) {
                PCLPointField field_custom_attr;
                field_custom_attr.name = kv.first;
                field_custom_attr.count = 1;
                SetPCDHeaderFieldTypeAndSizeFromDtype(kv.second.GetDtype(),
                                                      field_custom_attr);
                header.fields.push_back(field_custom_attr);

                header.elementnum++;
                header.pointsize += field_custom_attr.size;
            } else {
                utility::LogWarning(
                        "Write PCD : Skipping {} attribute. PointCloud "
                        "contains {} attribute which is not supported by PCD "
                        "IO. Only points, normals, colors and attributes with "
                        "shape (num_points, 1) are supported. Expected shape: "
                        "{} but got {}.",
                        kv.first, kv.first,
                        core::SizeVector(
                                {pointcloud.GetPointPositions().GetLength(), 1})
                                .ToString(),
                        kv.second.GetShape().ToString());
            }
        }
    }

    if (write_ascii) {
        header.datatype = PCD_DATA_ASCII;
    } else {
        if (compressed) {
            header.datatype = PCD_DATA_BINARY_COMPRESSED;
        } else {
            header.datatype = PCD_DATA_BINARY;
        }
    }
    return true;
}

bool WritePCDHeader(FILE *file, const PCDHeader &header) {
    fprintf(file, "# .PCD v%s - Point Cloud Data file format\n",
            header.version.c_str());
    fprintf(file, "VERSION %s\n", header.version.c_str());
    fprintf(file, "FIELDS");
    for (const auto &field : header.fields) {
        fprintf(file, " %s", field.name.c_str());
    }
    fprintf(file, "\n");
    fprintf(file, "SIZE");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.size);
    }
    fprintf(file, "\n");
    fprintf(file, "TYPE");
    for (const auto &field : header.fields) {
        fprintf(file, " %c", field.type);
    }
    fprintf(file, "\n");
    fprintf(file, "COUNT");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.count);
    }
    fprintf(file, "\n");
    fprintf(file, "WIDTH %d\n", header.width);
    fprintf(file, "HEIGHT %d\n", header.height);
    fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
    fprintf(file, "POINTS %d\n", header.points);

    switch (header.datatype) {
        case PCD_DATA_BINARY:
            fprintf(file, "DATA binary\n");
            break;
        case PCD_DATA_BINARY_COMPRESSED:
            fprintf(file, "DATA binary_compressed\n");
            break;
        case PCD_DATA_ASCII:
        default:
            fprintf(file, "DATA ascii\n");
            break;
    }
    return true;
}

struct WriteAttributePtr {
    WriteAttributePtr(const std::string &name,
                      const core::Dtype &dtype,
                      const void *data_ptr,
                      const int &group_size)
        : name_(name),
          dtype_(dtype),
          data_ptr_(data_ptr),
          group_size_(group_size) {}

    const std::string name_;
    const core::Dtype dtype_;
    const void *data_ptr_;
    const int group_size_;
};

template <typename scalar_t>
void ColorToUint8(const scalar_t *input_color, uint8_t *output_color) {
    utility::LogError(
            "Color format not supported. Supported color format includes "
            "Float32, Float64, UInt8, UInt16, UInt32.");
}

template <>
void ColorToUint8<float>(const float *input_color, uint8_t *output_color) {
    const float normalisation_factor =
            static_cast<float>(std::numeric_limits<uint8_t>::max());
    for (int i = 0; i < 3; ++i) {
        output_color[i] = static_cast<uint8_t>(
                std::round(std::min(1.f, std::max(0.f, input_color[2 - i])) *
                           normalisation_factor));
    }
    output_color[3] = 0;
}

template <>
void ColorToUint8<double>(const double *input_color, uint8_t *output_color) {
    const double normalisation_factor =
            static_cast<double>(std::numeric_limits<uint8_t>::max());
    for (int i = 0; i < 3; ++i) {
        output_color[i] = static_cast<uint8_t>(
                std::round(std::min(1., std::max(0., input_color[2 - i])) *
                           normalisation_factor));
    }
    output_color[3] = 0;
}

template <>
void ColorToUint8<uint8_t>(const uint8_t *input_color, uint8_t *output_color) {
    for (int i = 0; i < 3; ++i) {
        output_color[i] = input_color[2 - i];
    }
    output_color[3] = 0;
}

template <>
void ColorToUint8<uint16_t>(const uint16_t *input_color,
                            uint8_t *output_color) {
    const uint16_t normalisation_factor =
            static_cast<uint16_t>(std::numeric_limits<uint8_t>::max() /
                                  std::numeric_limits<uint16_t>::max());
    for (int i = 0; i < 3; ++i) {
        output_color[i] = input_color[2 - i] * normalisation_factor;
    }
    output_color[3] = 0;
}

template <>
void ColorToUint8<uint32_t>(const uint32_t *input_color,
                            uint8_t *output_color) {
    const uint32_t normalisation_factor =
            static_cast<uint32_t>(std::numeric_limits<uint8_t>::max() /
                                  std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < 3; ++i) {
        output_color[i] = input_color[2 - i] * normalisation_factor;
    }
    output_color[3] = 0;
}

core::Tensor PackColorsToFloat(const core::Tensor &colors_contiguous) {
    core::Tensor packed_color =
            core::Tensor::Empty({colors_contiguous.GetLength(), 1},
                                core::Float32, core::Device("CPU:0"));
    auto packed_color_ptr = packed_color.GetDataPtr<float>();

    DISPATCH_DTYPE_TO_TEMPLATE(colors_contiguous.GetDtype(), [&]() {
        auto colors_ptr = colors_contiguous.GetDataPtr<scalar_t>();
        core::ParallelFor(core::Device("CPU:0"), colors_contiguous.GetLength(),
                          [&](int64_t workload_idx) {
                              std::uint8_t rgba[4] = {0};
                              ColorToUint8<scalar_t>(
                                      colors_ptr + 3 * workload_idx, rgba);
                              float val = 0;
                              memcpy(&val, rgba, 4);
                              packed_color_ptr[workload_idx] = val;
                          });
    });

    return packed_color;
}

template <typename scalar_t>
int WriteElementDataToFileASCII(const scalar_t &data, FILE *file) {
    utility::LogError(
            "Color format not supported. Supported color format includes "
            "Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, "
            "UInt32, UInt64.");
}

template <>
int WriteElementDataToFileASCII<float>(const float &data, FILE *file) {
    return fprintf(file, "%.10g ", data);
}

template <>
int WriteElementDataToFileASCII<double>(const double &data, FILE *file) {
    return fprintf(file, "%.10g ", data);
}

template <>
int WriteElementDataToFileASCII<int8_t>(const int8_t &data, FILE *file) {
    return fprintf(file, "%.10d ", data);
}

template <>
int WriteElementDataToFileASCII<int16_t>(const int16_t &data, FILE *file) {
    return fprintf(file, "%.10hd ", data);
}

template <>
int WriteElementDataToFileASCII<int32_t>(const int32_t &data, FILE *file) {
    return fprintf(file, "%.10d ", data);
}

template <>
int WriteElementDataToFileASCII<int64_t>(const int64_t &data, FILE *file) {
    return fprintf(file, "%.10ld ", data);
}

template <>
int WriteElementDataToFileASCII<uint8_t>(const uint8_t &data, FILE *file) {
    return fprintf(file, "%.10ud ", data);
}

template <>
int WriteElementDataToFileASCII<uint16_t>(const uint16_t &data, FILE *file) {
    return fprintf(file, "%.10uhd ", data);
}

template <>
int WriteElementDataToFileASCII<uint32_t>(const uint32_t &data, FILE *file) {
    return fprintf(file, "%.10ud ", data);
}

template <>
int WriteElementDataToFileASCII<uint64_t>(const uint64_t &data, FILE *file) {
    return fprintf(file, "%.10lud ", data);
}

template <typename scalar_t>
size_t WriteElementDataToFileBIN(const scalar_t &data, FILE *file) {
    return fwrite(&data, sizeof(scalar_t), 1, file);
}

bool WritePCDData(FILE *file,
                  const PCDHeader &header,
                  const geometry::PointCloud &pointcloud,
                  const WritePointCloudOption &params) {
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write PLY failed: point cloud has 0 points.");
        return false;
    }

    // TODO: Add device transfer in tensor map and use it here.
    geometry::TensorMap t_map(pointcloud.GetPointAttr().Contiguous());
    int64_t num_points =
            static_cast<int64_t>(pointcloud.GetPointPositions().GetLength());

    std::vector<WriteAttributePtr> attribute_ptrs;

    attribute_ptrs.emplace_back("positions", t_map["positions"].GetDtype(),
                                t_map["positions"].GetDataPtr(), 3);

    if (pointcloud.HasPointNormals()) {
        attribute_ptrs.emplace_back("normals", t_map["normals"].GetDtype(),
                                    t_map["normals"].GetDataPtr(), 3);
    }

    if (pointcloud.HasPointColors()) {
        t_map["colors"] = PackColorsToFloat(t_map["colors"]);
        attribute_ptrs.emplace_back("colors", core::Float32,
                                    t_map["colors"].GetDataPtr(), 1);
    }

    // Sanity check for the attributes is done before adding it to the
    // `header.fields`.
    for (auto &field : header.fields) {
        if (field.name != "x" && field.name != "y" && field.name != "z" &&
            field.name != "normal_x" && field.name != "normal_y" &&
            field.name != "normal_z" && field.name != "rgb" &&
            field.name != "rgba" && field.count == 1) {
            attribute_ptrs.emplace_back(field.name,
                                        t_map[field.name].GetDtype(),
                                        t_map[field.name].GetDataPtr(), 1);
        }
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_points);
    if (header.datatype == PCD_DATA_ASCII) {
        for (int64_t i = 0; i < num_points; ++i) {
            for (auto &it : attribute_ptrs) {
                DISPATCH_DTYPE_TO_TEMPLATE(it.dtype_, [&]() {
                    const scalar_t *data_ptr =
                            static_cast<const scalar_t *>(it.data_ptr_);

                    for (int idx_offset = it.group_size_ * i;
                         idx_offset < it.group_size_ * (i + 1); ++idx_offset) {
                        WriteElementDataToFileASCII<scalar_t>(
                                data_ptr[idx_offset], file);
                    }
                });
            }

            fprintf(file, "\n");
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::vector<char> buffer_t;
        buffer_t.reserve(header.elementnum * header.points);

        for (int64_t i = 0; i < num_points; ++i) {
            for (auto &it : attribute_ptrs) {
                DISPATCH_DTYPE_TO_TEMPLATE(it.dtype_, [&]() {
                    const scalar_t *data_ptr =
                            static_cast<const scalar_t *>(it.data_ptr_);

                    for (int idx_offset = it.group_size_ * i;
                         idx_offset < it.group_size_ * (i + 1); ++idx_offset) {
                        const char *rhs_ptr = reinterpret_cast<const char *>(
                                &data_ptr[idx_offset]);
                        for (size_t byte = 0; byte < sizeof(scalar_t); byte++) {
                            char val = *(rhs_ptr + byte);
                            buffer_t.push_back(val);
                        }
                    }
                });
            }

            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }

        fwrite(buffer_t.data(), sizeof(char), buffer_t.size(), file);
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        double report_total = double(num_points * 2);
        // 0%-50% packing into buffer
        // 50%-75% compressing buffer
        // 75%-100% writing compressed buffer
        reporter.SetTotal(int(report_total));
        int strip_size = header.points;
        std::uint32_t buffer_size =
                (std::uint32_t)(header.elementnum * header.points);
        std::unique_ptr<float[]> buffer(new float[buffer_size]);
        std::unique_ptr<float[]> buffer_compressed(new float[buffer_size * 2]);
        for (int64_t i = 0; i < num_points; ++i) {
            utility::LogError("Unimplemented");

            int idx = 0;
            for (auto &it : attribute_ptrs) {
                DISPATCH_DTYPE_TO_TEMPLATE(it.dtype_, [&]() {
                    const scalar_t *data_ptr =
                            static_cast<const scalar_t *>(it.data_ptr_);

                    for (int idx_offset = it.group_size_ * i;
                         idx_offset < it.group_size_ * (i + 1); ++idx_offset) {
                        // TODO : This might not be correct to read. Find
                        // alternative to dump scalar_t type in buffer instead
                        // of type casting to float.
                        buffer[idx * strip_size + i] =
                                static_cast<float>(data_ptr[idx_offset]);
                    }
                });
            }

            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }

        std::uint32_t buffer_size_in_bytes = buffer_size * sizeof(float);
        std::uint32_t size_compressed =
                lzf_compress(buffer.get(), buffer_size_in_bytes,
                             buffer_compressed.get(), buffer_size_in_bytes * 2);
        if (size_compressed == 0) {
            utility::LogWarning("[WritePCDData] Failed to compress data.");
            return false;
        }
        utility::LogDebug(
                "[WritePCDData] {:d} bytes data compressed into {:d} bytes.",
                buffer_size_in_bytes, size_compressed);
        reporter.Update(int(report_total * 0.75));
        fwrite(&size_compressed, sizeof(size_compressed), 1, file);
        fwrite(&buffer_size_in_bytes, sizeof(buffer_size_in_bytes), 1, file);
        fwrite(buffer_compressed.get(), 1, size_compressed, file);
    }
    reporter.Finish();
    return true;
}

}  // unnamed namespace

bool ReadPointCloudFromPCD(const std::string &filename,
                           t::geometry::PointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    PCDHeader header;
    FILE *file = utility::filesystem::FOpen(filename.c_str(), "rb");
    if (file == NULL) {
        utility::LogWarning("Read PCD failed: unable to open file: {}",
                            filename);
        return false;
    }

    if (!ReadPCDHeader(file, header)) {
        utility::LogWarning("Read PCD failed: unable to parse header.");
        fclose(file);
        return false;
    }
    utility::LogDebug(
            "PCD header indicates {:d} fields, {:d} bytes per point, and {:d} "
            "points in total.",
            (int)header.fields.size(), header.pointsize, header.points);
    for (const auto &field : header.fields) {
        utility::LogDebug("{}, {}, {:d}, {:d}, {:d}", field.name.c_str(),
                          field.type, field.size, field.count, field.offset);
    }
    utility::LogDebug("Compression method is {:d}.", (int)header.datatype);
    utility::LogDebug("Points: {};  normals: {};  colors: {}",
                      header.has_attr["positions"] ? "yes" : "no",
                      header.has_attr["normals"] ? "yes" : "no",
                      header.has_attr["colors"] ? "yes" : "no");
    if (!ReadPCDData(file, header, pointcloud, params)) {
        utility::LogWarning("Read PCD failed: unable to read data.");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

bool WritePointCloudToPCD(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    PCDHeader header;
    if (!GenerateHeader(pointcloud, bool(params.write_ascii),
                        bool(params.compressed), header)) {
        utility::LogWarning("Write PCD failed: unable to generate header.");
        return false;
    }
    FILE *file = utility::filesystem::FOpen(filename.c_str(), "wb");
    if (file == NULL) {
        utility::LogWarning("Write PCD failed: unable to open file.");
        return false;
    }
    if (!WritePCDHeader(file, header)) {
        utility::LogWarning("Write PCD failed: unable to write header.");
        fclose(file);
        return false;
    }
    if (!WritePCDData(file, header, pointcloud, params)) {
        utility::LogWarning("Write PCD failed: unable to write data.");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d

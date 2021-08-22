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
    utility::LogInfo("Checking Header");

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
                 i++, count_offset += 1, offset += 4) {
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
                 i++, offset += col_type) {
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
            for (size_t i = 0; i < specified_channel_count; i++) {
                header.fields[i].type = st[i + 1].c_str()[0];
            }
        } else if (line_type.substr(0, 5) == "COUNT") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            int count_offset = 0, offset = 0, col_count = 0;
            for (size_t i = 0; i < specified_channel_count; i++) {
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

struct AttributePtr {
    AttributePtr(void *data_ptr = nullptr,
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

inline void ReadASCIIPCDColorsFromField(AttributePtr &attr,
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

inline void ReadBinaryPCDColorsFromField(AttributePtr &attr,
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

inline void ReadPCDElementsFromField(AttributePtr &attr,
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

    std::unordered_map<std::string, AttributePtr> map_field_to_attr_ptr;

    if (header.has_attr["positions"]) {
        pointcloud.SetPointPositions(core::Tensor::Empty(
                {header.points, 3}, header.attr_dtype["positions"]));

        void *data_ptr = pointcloud.GetPointPositions().GetDataPtr();
        AttributePtr position_x(data_ptr, 0, 3, header.points);
        AttributePtr position_y(data_ptr, 1, 3, header.points);
        AttributePtr position_z(data_ptr, 2, 3, header.points);

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
        AttributePtr normal_x(data_ptr, 0, 3, header.points);
        AttributePtr normal_y(data_ptr, 1, 3, header.points);
        AttributePtr normal_z(data_ptr, 2, 3, header.points);

        map_field_to_attr_ptr.emplace(std::string("normal_x"), normal_x);
        map_field_to_attr_ptr.emplace(std::string("normal_y"), normal_y);
        map_field_to_attr_ptr.emplace(std::string("normal_z"), normal_z);
    }
    if (header.has_attr["colors"]) {
        // Colors stored in a PCD file is ALWAYS in UInt8 format.
        // However it is stored as a single packed value, which itself maybe of
        // Int8, UInt8 or Float32 in ASCII and Float32 in Binary format.
        pointcloud.SetPointColors(
                core::Tensor::Empty({header.points, 3}, core::UInt8));

        void *data_ptr = pointcloud.GetPointColors().GetDataPtr();
        AttributePtr colors(data_ptr, 0, 3, header.points);

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
            AttributePtr attr(data_ptr, 0, 1, header.points);

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
            for (size_t i = 0; i < header.fields.size(); i++) {
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
        for (int i = 0; i < header.points; i++) {
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
                for (int i = 0; i < header.points; i++) {
                    ReadBinaryPCDColorsFromField(
                            map_field_to_attr_ptr["colors"], field,
                            base_ptr + i * field.size * field.count, i);
                }
            } else {
                for (int i = 0; i < header.points; i++) {
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

}  // namespace io
}  // namespace t
}  // namespace open3d

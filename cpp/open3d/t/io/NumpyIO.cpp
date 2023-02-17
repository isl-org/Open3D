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

// Contains source code from: https://github.com/rogersce/cnpy.
//
// The MIT License
//
// Copyright (c) Carl Rogers, 2011
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "open3d/t/io/NumpyIO.h"

#include <zlib.h>

#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/core/Blob.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

class CharVector {
public:
    CharVector() {}
    CharVector(size_t size) : buffer_(size) {}
    CharVector(size_t size, const char value) : buffer_(size, value) {}

    template <typename T>
    CharVector& Append(const T& rhs) {
        // Write in little endian.
        const char* rhs_ptr = reinterpret_cast<const char*>(&rhs);
        buffer_.reserve(buffer_.size() + sizeof(T));
        for (size_t byte = 0; byte < sizeof(T); byte++) {
            char val = *(rhs_ptr + byte);
            buffer_.push_back(val);
        }
        return *this;
    }

    CharVector& Append(const std::string& rhs) {
        buffer_.insert(buffer_.end(), rhs.begin(), rhs.end());
        return *this;
    }

    CharVector& Append(const char* rhs) {
        // Write in little endian.
        size_t len = strlen(rhs);
        buffer_.reserve(buffer_.size() + len);
        for (size_t byte = 0; byte < len; byte++) {
            buffer_.push_back(rhs[byte]);
        }
        return *this;
    }

    template <typename InputIt>
    CharVector& Append(InputIt first, InputIt last) {
        buffer_.insert(buffer_.end(), first, last);
        return *this;
    }

    template <typename T>
    CharVector& Append(size_t count, const T& value) {
        for (size_t i = 0; i < count; ++i) {
            Append(value);
        }
        return *this;
    }

    CharVector& Append(const CharVector& other) {
        buffer_.insert(buffer_.end(), other.buffer_.begin(),
                       other.buffer_.end());
        return *this;
    }

    std::vector<char>::iterator Begin() { return buffer_.begin(); }

    std::vector<char>::iterator End() { return buffer_.end(); }

    size_t Size() const { return buffer_.size(); }

    void Resize(size_t count) { buffer_.resize(count); }

    char* Data() { return buffer_.data(); }

    const char* Data() const { return buffer_.data(); }

    char& operator[](size_t index) { return buffer_[index]; }

    const char& operator[](size_t index) const { return buffer_[index]; }

private:
    std::vector<char> buffer_;
};

static char BigEndianChar() {
    int x = 1;
    return ((reinterpret_cast<char*>(&x))[0]) ? '<' : '>';
}

static char DtypeToChar(const core::Dtype& dtype) {
    // Not all dtypes are supported.
    // 'f': float, double, long double
    // 'i': int, char, short, long, long long
    // 'u': unsigned char, unsigned short, unsigned long, unsigned long long,
    //      unsigned int
    // 'b': bool
    // 'c': std::complex<float>, std::complex<double>),
    //      std::complex<long double>)
    // '?': object
    if (dtype == core::Float32) return 'f';
    if (dtype == core::Float64) return 'f';
    if (dtype == core::Int8) return 'i';
    if (dtype == core::Int16) return 'i';
    if (dtype == core::Int32) return 'i';
    if (dtype == core::Int64) return 'i';
    if (dtype == core::UInt8) return 'u';
    if (dtype == core::UInt16) return 'u';
    if (dtype == core::UInt32) return 'u';
    if (dtype == core::UInt64) return 'u';
    if (dtype == core::Bool) return 'b';
    utility::LogError("Unsupported dtype: {}", dtype.ToString());
    return '\0';
}

static CharVector CreateNumpyHeader(const core::SizeVector& shape,
                                    const core::Dtype& dtype) {
    // {}     -> "()"
    // {1}    -> "(1,)"
    // {1, 2} -> "(1, 2)"
    std::stringstream shape_ss;
    if (shape.size() == 0) {
        shape_ss << "()";
    } else if (shape.size() == 1) {
        shape_ss << fmt::format("({},)", shape[0]);
    } else {
        shape_ss << "(";
        shape_ss << shape[0];
        for (size_t i = 1; i < shape.size(); i++) {
            shape_ss << ", ";
            shape_ss << shape[i];
        }
        if (shape.size() == 1) {
            shape_ss << ",";
        }
        shape_ss << ")";
    }

    // Pad with spaces so that preamble+dict is modulo 16 bytes.
    // - Preamble is 10 bytes.
    // - Dict needs to end with '\n'.
    // - Header dict size includes the padding size and '\n'.
    std::string property_dict_body = fmt::format(
            "{{'descr': '{}{}{}', 'fortran_order': False, 'shape': {}, }}",
            BigEndianChar(), DtypeToChar(dtype), dtype.ByteSize(),
            shape_ss.str());

    CharVector property_dict;
    property_dict.Append(property_dict_body);
    // {0, 1, ..., 15}
    size_t padding_count = 16 - (10 + property_dict.Size()) % 16 - 1;
    property_dict.Append(padding_count, ' ');
    property_dict.Append('\n');

    CharVector header;
    header.Append<unsigned char>(0x93);  // Magic value
    header.Append("NUMPY");              // Magic value
    header.Append<uint8_t>(0x01);        // Major version
    header.Append<uint8_t>(0x00);        // Minor version
    header.Append<uint16_t>(property_dict.Size());
    header.Append(property_dict);

    return header;  // Use move since CharVector is inherited.
}

static std::tuple<core::SizeVector, char, int64_t, bool> ParsePropertyDict(
        const std::string& header) {
    core::SizeVector shape;
    char type;
    int64_t word_size;
    bool fortran_order;

    size_t loc1;
    size_t loc2;

    // Fortran order.
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos) {
        utility::LogError("Failed to find header keyword: 'fortran_order'");
    }
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // Shape.
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos) {
        utility::LogError("Failed to find header keyword: '(' or ')'");
    }

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // Endian, word size, data type.
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array.
    loc1 = header.find("descr");
    if (loc1 == std::string::npos) {
        utility::LogError("Failed to find header keyword: 'descr'");
    }

    loc1 += 9;
    bool little_endian =
            (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    if (!little_endian) {
        utility::LogError("Only big endian is supported.");
    }

    type = header[loc1 + 1];

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());

    return std::make_tuple(shape, type, word_size, fortran_order);
}

// Returns header length, which is the length of the string of property dict.
// The preamble must be at least 10 bytes.
// Ref: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
//
// - bytes[0] to bytes[5]             : \x93NUMPY # Magic string
// - bytes[6]                         : \x01      # Major version, unsigned
// - bytes[7]                         : \x00      # Minor version, unsigned
// - bytes[8] to bytes[9]             : HEADER_LEN little-endian uint16_t
// - bytes[10] to bytes[10+HEADER_LEN]: Dict, padded, terminated by '\n'
// - (10 + HEADER_LEN) % 64 == 0      : Guaranteed
//
// - We only support Version 1.0 for now.
// - Version 2.0+ supports up to 4GiB HEADER_LEN and the HEADER_LEN is
//   replaced from uint16_t to uint32_t.
// - Version 3.0 uses utf8-encoded header string.
static size_t ParseNpyPreamble(const char* preamble) {
    if (*reinterpret_cast<const unsigned char*>(&preamble[0]) !=
                static_cast<unsigned char>(0x93) ||
        preamble[1] != 'N' || preamble[2] != 'U' || preamble[3] != 'M' ||
        preamble[4] != 'P' || preamble[5] != 'Y') {
        utility::LogError("Invalid Numpy preamble {}{}{}{}{}{}.", preamble[0],
                          preamble[1], preamble[2], preamble[3], preamble[4],
                          preamble[5]);
    }
    if (preamble[6] != static_cast<uint8_t>(0x01) ||
        preamble[7] != static_cast<uint8_t>(0x00)) {
        utility::LogError(
                "Not supported Numpy format version: {}.{}. Only version 1.0 "
                "is supported.",
                preamble[6], preamble[7]);
    }
    uint16_t header_len = *reinterpret_cast<const uint16_t*>(&preamble[8]);
    return static_cast<size_t>(header_len);
}

// Returns {shape, type(char), word_size, fortran_order}.
// This will advance the file pointer to the end of the header.
static std::tuple<core::SizeVector, char, int64_t, bool> ParseNpyHeaderFromFile(
        FILE* fp) {
    const size_t preamble_len = 10;  // Version 1.0 assumed.
    CharVector preamble(preamble_len);
    if (fread(preamble.Data(), sizeof(char), preamble_len, fp) !=
        preamble_len) {
        utility::LogError("Header preamble cannot be read.");
    }
    const size_t header_len = ParseNpyPreamble(preamble.Data());

    CharVector header(header_len, 0);
    if (fread(header.Data(), sizeof(char), header_len, fp) != header_len) {
        utility::LogError("Failed to read header dictionary.");
    }
    if (header[header_len - 1] != '\n') {
        utility::LogError("Numpy header not terminated by null character.");
    }

    return ParsePropertyDict(std::string(header.Data(), header_len));
}

static std::tuple<core::SizeVector, char, int64_t, bool>
ParseNpyHeaderFromBuffer(const char* buffer) {
    const size_t header_len = ParseNpyPreamble(buffer);
    std::string header(reinterpret_cast<const char*>(buffer + 10), header_len);
    return ParsePropertyDict(header);
}

static std::tuple<size_t, size_t, size_t> ParseZipFooter(FILE* fp) {
    size_t footer_len = 22;
    CharVector footer(footer_len);
    fseek(fp, -static_cast<int64_t>(footer_len), SEEK_END);
    if (fread(footer.Data(), sizeof(char), footer_len, fp) != footer_len) {
        utility::LogError("Footer fread failed.");
    }

    // clang-format off
    uint16_t disk_no              = *reinterpret_cast<uint16_t*>(&footer[4 ]);
    uint16_t disk_start           = *reinterpret_cast<uint16_t*>(&footer[6 ]);
    uint16_t nrecs_on_disk        = *reinterpret_cast<uint16_t*>(&footer[8 ]);
    uint16_t nrecs                = *reinterpret_cast<uint16_t*>(&footer[10]);
    uint32_t global_header_size   = *reinterpret_cast<uint32_t*>(&footer[12]);
    uint32_t global_header_offset = *reinterpret_cast<uint32_t*>(&footer[16]);
    uint16_t comment_len          = *reinterpret_cast<uint16_t*>(&footer[20]);
    // clang-format on

    if (disk_no != 0 || disk_start != 0 || comment_len != 0) {
        utility::LogError("Unsupported zip footer.");
    }
    if (nrecs_on_disk != nrecs) {
        utility::LogError("Unsupported zip footer.");
    }

    return std::make_tuple(static_cast<size_t>(nrecs), global_header_size,
                           global_header_offset);
}

static void WriteNpzOneTensor(const std::string& file_name,
                              const std::string& tensor_name,
                              const core::Tensor& tensor,
                              bool append) {
    const void* data = tensor.GetDataPtr();
    const core::SizeVector shape = tensor.GetShape();
    const core::Dtype dtype = tensor.GetDtype();
    const int64_t element_byte_size = dtype.ByteSize();

    utility::filesystem::CFile cfile;
    const std::string mode = append ? "r+b" : "wb";
    if (!cfile.Open(file_name, mode)) {
        utility::LogError("Failed to open file {}, error: {}.", file_name,
                          cfile.GetError());
    }
    FILE* fp = cfile.GetFILE();

    size_t nrecs = 0;
    size_t global_header_offset = 0;
    CharVector global_header;

    if (append) {
        // Zip file exists. we need to add a new npy file to it. First read the
        // footer. This gives us the offset and size of the global header then
        // read and store the global header. Below, we will write the the new
        // data at the start of the global header then append the global header
        // and footer below it.
        size_t global_header_size;
        std::tie(nrecs, global_header_size, global_header_offset) =
                ParseZipFooter(fp);
        fseek(fp, global_header_offset, SEEK_SET);
        global_header.Resize(global_header_size);
        size_t res = fread(global_header.Data(), sizeof(char),
                           global_header_size, fp);
        if (res != global_header_size) {
            utility::LogError("Header read error while saving to npz.");
        }
        fseek(fp, global_header_offset, SEEK_SET);
    }

    CharVector npy_header = CreateNumpyHeader(shape, dtype);

    size_t nels = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<size_t>());
    size_t nbytes = nels * element_byte_size + npy_header.Size();

    // Get the CRC of the data to be added.
    uint32_t crc = crc32(0L, reinterpret_cast<uint8_t*>(npy_header.Data()),
                         npy_header.Size());
    crc = crc32(crc, static_cast<const uint8_t*>(data),
                nels * element_byte_size);

    // The ".npy" suffix will be removed when npz is read.
    std::string var_name = tensor_name + ".npy";

    // Build the local header.
    CharVector local_header;
    local_header.Append("PK");                       // First part of sig
    local_header.Append<uint16_t>(0x0403);           // Second part of sig
    local_header.Append<uint16_t>(20);               // Min version to extract
    local_header.Append<uint16_t>(0);                // General purpose bit flag
    local_header.Append<uint16_t>(0);                // Compression method
    local_header.Append<uint16_t>(0);                // File last mod time
    local_header.Append<uint16_t>(0);                // File last mod date
    local_header.Append<uint32_t>(crc);              // CRC
    local_header.Append<uint32_t>(nbytes);           // Compressed size
    local_header.Append<uint32_t>(nbytes);           // Uncompressed size
    local_header.Append<uint16_t>(var_name.size());  // Varaible's name length
    local_header.Append<uint16_t>(0);                // Extra field length
    local_header.Append(var_name);

    // Build global header.
    global_header.Append("PK");              // First part of sig
    global_header.Append<uint16_t>(0x0201);  // Second part of sig
    global_header.Append<uint16_t>(20);      // Version made by
    global_header.Append(local_header.Begin() + 4, local_header.Begin() + 30);
    global_header.Append<uint16_t>(0);  // File comment length
    global_header.Append<uint16_t>(0);  // Disk number where file starts
    global_header.Append<uint16_t>(0);  // Internal file attributes
    global_header.Append<uint32_t>(0);  // External file attributes
    // Relative offset of local file header, since it begins where the global
    // header used to begin.
    global_header.Append<uint32_t>(global_header_offset);
    global_header.Append(var_name);

    // Build footer.
    CharVector footer;
    footer.Append("PK");                 // First part of sig
    footer.Append<uint16_t>(0x0605);     // Second part of sig
    footer.Append<uint16_t>(0);          // Number of this disk
    footer.Append<uint16_t>(0);          // Disk where footer starts
    footer.Append<uint16_t>(nrecs + 1);  // Number of records on this disk
    footer.Append<uint16_t>(nrecs + 1);  // Total number of records
    footer.Append<uint32_t>(global_header.Size());  // Nbytes of global headers
    // Offset of start of global headers, since global header now starts after
    // newly written array.
    footer.Append<uint32_t>(global_header_offset + nbytes +
                            local_header.Size());
    footer.Append<uint16_t>(0);  // Zip file comment length.

    // Write everything.
    fwrite(local_header.Data(), sizeof(char), local_header.Size(), fp);
    fwrite(npy_header.Data(), sizeof(char), npy_header.Size(), fp);
    fwrite(data, element_byte_size, nels, fp);
    fwrite(global_header.Data(), sizeof(char), global_header.Size(), fp);
    fwrite(footer.Data(), sizeof(char), footer.Size(), fp);
}

static void WriteNpzEmpty(const std::string& file_name) {
    utility::filesystem::CFile cfile;
    if (!cfile.Open(file_name, "wb")) {
        utility::LogError("Failed to open file {}, error: {}.", file_name,
                          cfile.GetError());
    }
    FILE* fp = cfile.GetFILE();

    // Build footer.
    CharVector footer;
    footer.Append("PK");              // First part of sig
    footer.Append<uint16_t>(0x0605);  // Second part of sig
    footer.Append<uint16_t>(0);       // Number of this disk
    footer.Append<uint16_t>(0);       // Disk where footer starts
    footer.Append<uint16_t>(0);       // Number of records on this disk
    footer.Append<uint16_t>(0);       // Total number of records
    footer.Append<uint32_t>(0);       // Nbytes of global headers
    footer.Append<uint32_t>(0);       // External file attributes
    footer.Append<uint16_t>(0);       // Zip file comment length.
    if (footer.Size() != 22) {
        utility::LogError("Internal error: empty zip file must have size 22.");
    }

    // Write everything.
    fwrite(footer.Data(), sizeof(char), footer.Size(), fp);
}

class NumpyArray {
public:
    NumpyArray(const core::Tensor& t)
        : shape_(t.GetShape()),
          type_(DtypeToChar(t.GetDtype())),
          word_size_(t.GetDtype().ByteSize()),
          fortran_order_(false) {
        blob_ = t.To(core::Device("CPU:0")).Contiguous().GetBlob();
    }

    NumpyArray(const core::SizeVector& shape,
               char type,
               int64_t word_size,
               bool fortran_order)
        : shape_(shape),
          type_(type),
          word_size_(word_size),
          fortran_order_(fortran_order) {
        blob_ = std::make_shared<core::Blob>(NumBytes(), core::Device("CPU:0"));
    }

    template <typename T>
    T* GetDataPtr() {
        return reinterpret_cast<T*>(blob_->GetDataPtr());
    }

    template <typename T>
    const T* GetDataPtr() const {
        return reinterpret_cast<const T*>(blob_->GetDataPtr());
    }

    core::Dtype GetDtype() const {
        if (type_ == 'f' && word_size_ == 4) return core::Float32;
        if (type_ == 'f' && word_size_ == 8) return core::Float64;
        if (type_ == 'i' && word_size_ == 1) return core::Int8;
        if (type_ == 'i' && word_size_ == 2) return core::Int16;
        if (type_ == 'i' && word_size_ == 4) return core::Int32;
        if (type_ == 'i' && word_size_ == 8) return core::Int64;
        if (type_ == 'u' && word_size_ == 1) return core::UInt8;
        if (type_ == 'u' && word_size_ == 2) return core::UInt16;
        if (type_ == 'u' && word_size_ == 4) return core::UInt32;
        if (type_ == 'u' && word_size_ == 8) return core::UInt64;
        if (type_ == 'b') return core::Bool;

        return core::Undefined;
    }

    core::SizeVector GetShape() const { return shape_; }

    bool IsFortranOrder() const { return fortran_order_; }

    int64_t NumBytes() const { return NumElements() * word_size_; }

    int64_t NumElements() const { return shape_.NumElements(); }

    core::Tensor ToTensor() const {
        if (fortran_order_) {
            utility::LogError("Cannot load Numpy array with fortran_order.");
        }
        core::Dtype dtype = GetDtype();
        if (dtype.GetDtypeCode() == core::Dtype::DtypeCode::Undefined) {
            utility::LogError(
                    "Cannot load Numpy array with Numpy dtype={} and "
                    "word_size={}.",
                    type_, word_size_);
        }
        // t.blob_ is the same as blob_, no need for memory copy.
        core::Tensor t(shape_, core::shape_util::DefaultStrides(shape_),
                       const_cast<void*>(GetDataPtr<void>()), dtype, blob_);
        return t;
    }

    void Save(std::string file_name) const {
        utility::filesystem::CFile cfile;
        if (!cfile.Open(file_name, "wb")) {
            utility::LogError("Failed to open file {}, error: {}.", file_name,
                              cfile.GetError());
        }
        FILE* fp = cfile.GetFILE();

        CharVector header = CreateNumpyHeader(shape_, GetDtype());
        fseek(fp, 0, SEEK_SET);
        fwrite(header.Data(), sizeof(char), header.Size(), fp);
        fseek(fp, 0, SEEK_END);
        fwrite(GetDataPtr<void>(), static_cast<size_t>(GetDtype().ByteSize()),
               static_cast<size_t>(shape_.NumElements()), fp);
    }

private:
    std::shared_ptr<core::Blob> blob_ = nullptr;
    core::SizeVector shape_;
    char type_;
    int64_t word_size_;
    bool fortran_order_;
};

static NumpyArray CreateNumpyArrayFromFile(FILE* fp) {
    if (!fp) {
        utility::LogError("Unable to open file ptr.");
    }

    core::SizeVector shape;
    char type;
    int64_t word_size;
    bool fortran_order;
    std::tie(shape, type, word_size, fortran_order) =
            ParseNpyHeaderFromFile(fp);

    NumpyArray arr(shape, type, word_size, fortran_order);
    size_t nread = fread(arr.GetDataPtr<char>(), 1,
                         static_cast<size_t>(arr.NumBytes()), fp);
    if (nread != static_cast<size_t>(arr.NumBytes())) {
        utility::LogError("Failed to read array data.");
    }
    return arr;
}

static NumpyArray CreateNumpyArrayFromCompressedFile(
        FILE* fp,
        uint32_t num_compressed_bytes,
        uint32_t num_uncompressed_bytes) {
    CharVector buffer_compressed(num_compressed_bytes);
    CharVector buffer_uncompressed(num_uncompressed_bytes);
    size_t nread = fread(buffer_compressed.Data(), 1, num_compressed_bytes, fp);
    if (nread != num_compressed_bytes) {
        utility::LogError("Failed to read compressed data.");
    }

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in = num_compressed_bytes;
    d_stream.next_in =
            reinterpret_cast<unsigned char*>(buffer_compressed.Data());
    d_stream.avail_out = num_uncompressed_bytes;
    d_stream.next_out =
            reinterpret_cast<unsigned char*>(buffer_uncompressed.Data());

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);
    if (err != Z_OK) {
        utility::LogError("Failed to decompress data.");
    }

    core::SizeVector shape;
    char type;
    size_t word_size;
    bool fortran_order;
    std::tie(shape, type, word_size, fortran_order) =
            ParseNpyHeaderFromBuffer(buffer_uncompressed.Data());

    NumpyArray array(shape, type, word_size, fortran_order);

    size_t offset = num_uncompressed_bytes - array.NumBytes();
    memcpy(array.GetDataPtr<char>(), buffer_uncompressed.Data() + offset,
           array.NumBytes());

    return array;
}

core::Tensor ReadNpy(const std::string& file_name) {
    utility::filesystem::CFile cfile;
    if (!cfile.Open(file_name, "rb")) {
        utility::LogError("Failed to open file {}, error: {}.", file_name,
                          cfile.GetError());
    }
    return CreateNumpyArrayFromFile(cfile.GetFILE()).ToTensor();
}

void WriteNpy(const std::string& file_name, const core::Tensor& tensor) {
    NumpyArray(tensor).Save(file_name);
}

std::unordered_map<std::string, core::Tensor> ReadNpz(
        const std::string& file_name) {
    utility::filesystem::CFile cfile;
    if (!cfile.Open(file_name, "rb")) {
        utility::LogError("Failed to open file {}, error: {}.", file_name,
                          cfile.GetError());
    }
    FILE* fp = cfile.GetFILE();

    std::unordered_map<std::string, core::Tensor> tensor_map;

    // It's possible to check tensor_name and only one selected numpy array,
    // here we load all of them.
    while (true) {
        CharVector local_header(30);
        size_t local_header_bytes =
                fread(local_header.Data(), sizeof(char), 30, fp);

        // An empty zip file has exactly 22 bytes.
        if (local_header_bytes == 22) {
            const char empty_zip_bytes[22] = {
                    0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            if (std::memcmp(empty_zip_bytes, local_header.Data(), 22) == 0) {
                break;
            } else {
                utility::LogError("Invalid empty .npz file.");
            }
        }

        if (local_header_bytes != 30) {
            utility::LogError("Failed to read local header in npz.");
        }

        // If we've reached the global header, stop reading.
        if (local_header[2] != 0x03 || local_header[3] != 0x04) {
            break;
        }

        // Read tensor name.
        uint16_t tensor_name_len =
                *reinterpret_cast<uint16_t*>(&local_header[26]);
        CharVector tensor_name_buf(tensor_name_len, ' ');
        if (fread(tensor_name_buf.Data(), sizeof(char), tensor_name_len, fp) !=
            tensor_name_len) {
            utility::LogError("Failed to read tensor name in npz.");
        }

        // Erase the trailing ".npy".
        std::string tensor_name(tensor_name_buf.Begin(), tensor_name_buf.End());
        tensor_name.erase(tensor_name.end() - 4, tensor_name.end());

        // Read extra field.
        uint16_t extra_field_len =
                *reinterpret_cast<uint16_t*>(&local_header[28]);
        if (extra_field_len > 0) {
            CharVector buff(extra_field_len);
            if (fread(buff.Data(), sizeof(char), extra_field_len, fp) !=
                extra_field_len) {
                utility::LogError("Failed to read extra field in npz.");
            }
        }

        uint16_t compressed_method =
                *reinterpret_cast<uint16_t*>(&local_header[8]);
        uint32_t num_compressed_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[18]);
        uint32_t num_uncompressed_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[22]);

        if (compressed_method == 0) {
            tensor_map[tensor_name] = CreateNumpyArrayFromFile(fp).ToTensor();
        } else {
            tensor_map[tensor_name] =
                    CreateNumpyArrayFromCompressedFile(fp, num_compressed_bytes,
                                                       num_uncompressed_bytes)
                            .ToTensor();
        }
    }

    return tensor_map;
}

void WriteNpz(const std::string& file_name,
              const std::unordered_map<std::string, core::Tensor>& tensor_map) {
    if (tensor_map.empty()) {
        WriteNpzEmpty(file_name);
    }

    std::unordered_map<std::string, core::Tensor> contiguous_tensor_map;
    for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it) {
        contiguous_tensor_map[it->first] =
                it->second.To(core::Device("CPU:0")).Contiguous();
    }

    // TODO: WriteNpzOneTensor is called multiple times in order to write
    // multiple tensors. This requires opening/closing the npz file for multiple
    // times, which is not optimal.
    // TODO: Support writing in compressed mode: np.savez_compressed().
    bool is_first_tensor = true;
    for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it) {
        core::Tensor tensor = it->second.To(core::Device("CPU:0")).Contiguous();
        if (is_first_tensor) {
            WriteNpzOneTensor(file_name, it->first, tensor, /*append=*/false);
            is_first_tensor = false;
        } else {
            WriteNpzOneTensor(file_name, it->first, tensor, /*append=*/true);
        }
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d

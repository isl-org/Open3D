// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dtype.h"

namespace open3d {
namespace core {

// clang-format off
static_assert(sizeof(float   ) == 4, "Unsupported platform: float must be 4 bytes."   );
static_assert(sizeof(double  ) == 8, "Unsupported platform: double must be 8 bytes."  );
static_assert(sizeof(int     ) == 4, "Unsupported platform: int must be 4 bytes."     );
static_assert(sizeof(int8_t  ) == 1, "Unsupported platform: int8_t must be 1 byte."   );
static_assert(sizeof(int16_t ) == 2, "Unsupported platform: int16_t must be 2 bytes." );
static_assert(sizeof(int32_t ) == 4, "Unsupported platform: int32_t must be 4 bytes." );
static_assert(sizeof(int64_t ) == 8, "Unsupported platform: int64_t must be 8 bytes." );
static_assert(sizeof(uint8_t ) == 1, "Unsupported platform: uint8_t must be 1 byte."  );
static_assert(sizeof(uint16_t) == 2, "Unsupported platform: uint16_t must be 2 bytes.");
static_assert(sizeof(uint32_t) == 4, "Unsupported platform: uint32_t must be 4 bytes.");
static_assert(sizeof(uint64_t) == 8, "Unsupported platform: uint64_t must be 8 bytes.");
static_assert(sizeof(bool    ) == 1, "Unsupported platform: bool must be 1 byte."     );

const Dtype Dtype::Undefined(Dtype::DtypeCode::Undefined, 1, "Undefined");
const Dtype Dtype::Float32  (Dtype::DtypeCode::Float,     4, "Float32"  );
const Dtype Dtype::Float64  (Dtype::DtypeCode::Float,     8, "Float64"  );
const Dtype Dtype::Int8     (Dtype::DtypeCode::Int,       1, "Int8"     );
const Dtype Dtype::Int16    (Dtype::DtypeCode::Int,       2, "Int16"    );
const Dtype Dtype::Int32    (Dtype::DtypeCode::Int,       4, "Int32"    );
const Dtype Dtype::Int64    (Dtype::DtypeCode::Int,       8, "Int64"    );
const Dtype Dtype::UInt8    (Dtype::DtypeCode::UInt,      1, "UInt8"    );
const Dtype Dtype::UInt16   (Dtype::DtypeCode::UInt,      2, "UInt16"   );
const Dtype Dtype::UInt32   (Dtype::DtypeCode::UInt,      4, "UInt32"   );
const Dtype Dtype::UInt64   (Dtype::DtypeCode::UInt,      8, "UInt64"   );
const Dtype Dtype::Bool     (Dtype::DtypeCode::Bool,      1, "Bool"     );
// clang-format on

const Dtype Undefined = Dtype::Undefined;
const Dtype Float32 = Dtype::Float32;
const Dtype Float64 = Dtype::Float64;
const Dtype Int8 = Dtype::Int8;
const Dtype Int16 = Dtype::Int16;
const Dtype Int32 = Dtype::Int32;
const Dtype Int64 = Dtype::Int64;
const Dtype UInt8 = Dtype::UInt8;
const Dtype UInt16 = Dtype::UInt16;
const Dtype UInt32 = Dtype::UInt32;
const Dtype UInt64 = Dtype::UInt64;
const Dtype Bool = Dtype::Bool;

Dtype::Dtype(DtypeCode dtype_code, int64_t byte_size, const std::string &name)
    : dtype_code_(dtype_code), byte_size_(byte_size) {
    if (name.size() > max_name_len_ - 1) {
        utility::LogError("Name {} must be shorter.", name);
    } else {
        std::strncpy(name_, name.c_str(), max_name_len_);
        name_[max_name_len_ - 1] = '\0';
    }
}

bool Dtype::operator==(const Dtype &other) const {
    return dtype_code_ == other.dtype_code_ && byte_size_ == other.byte_size_ &&
           std::strcmp(name_, other.name_) == 0;
}

bool Dtype::operator!=(const Dtype &other) const { return !(*this == other); }

}  // namespace core
}  // namespace open3d

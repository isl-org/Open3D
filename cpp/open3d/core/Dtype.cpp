
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

#include "open3d/core/Dtype.h"

namespace open3d {
namespace core {

static_assert(sizeof(float) == 4,
              "Unsupported platform: float must be 4 bytes.");
static_assert(sizeof(double) == 8,
              "Unsupported platform: double must be 8 bytes.");
static_assert(sizeof(int) == 4, "Unsupported platform: int must be 4 bytes.");
static_assert(sizeof(int32_t) == 4,
              "Unsupported platform: int32_t must be 4 bytes.");
static_assert(sizeof(int64_t) == 8,
              "Unsupported platform: int64_t must be 8 bytes.");
static_assert(sizeof(uint8_t) == 1,
              "Unsupported platform: uint8_t must be 1 byte.");
static_assert(sizeof(uint16_t) == 2,
              "Unsupported platform: uint16_t must be 2 bytes.");
static_assert(sizeof(bool) == 1, "Unsupported platform: bool must be 1 byte.");

// clang-format off
const Dtype Dtype::Undefined = Dtype(Dtype::DtypeCode::Undefined, 1, "Undefined");
const Dtype Dtype::Float32   = Dtype(Dtype::DtypeCode::Float,     4, "Float32"  );
const Dtype Dtype::Float64   = Dtype(Dtype::DtypeCode::Float,     8, "Float64"  );
const Dtype Dtype::Int32     = Dtype(Dtype::DtypeCode::Int,       4, "Int32"    );
const Dtype Dtype::Int64     = Dtype(Dtype::DtypeCode::Int,       8, "Int64"    );
const Dtype Dtype::UInt8     = Dtype(Dtype::DtypeCode::UInt,      1, "UInt8"    );
const Dtype Dtype::UInt16    = Dtype(Dtype::DtypeCode::UInt,      2, "UInt16"   );
const Dtype Dtype::Bool      = Dtype(Dtype::DtypeCode::Bool,      1, "Bool"     );

const Dtype Undefined = Dtype(Dtype::DtypeCode::Undefined, 1, "Undefined");
const Dtype Float32   = Dtype(Dtype::DtypeCode::Float,     4, "Float32"  );
const Dtype Float64   = Dtype(Dtype::DtypeCode::Float,     8, "Float64"  );
const Dtype Int32     = Dtype(Dtype::DtypeCode::Int,       4, "Int32"    );
const Dtype Int64     = Dtype(Dtype::DtypeCode::Int,       8, "Int64"    );
const Dtype UInt8     = Dtype(Dtype::DtypeCode::UInt,      1, "UInt8"    );
const Dtype UInt16    = Dtype(Dtype::DtypeCode::UInt,      2, "UInt16"   );
const Dtype Bool      = Dtype(Dtype::DtypeCode::Bool,      1, "Bool"     );
// clang-format on

}  // namespace core
}  // namespace open3d

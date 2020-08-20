
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

// clang-format off
const Dtype Dtype::Undefined = Dtype(Dtype::DtypeCode::Undefined, 1, "Undefined");
const Dtype Dtype::Float32   = Dtype(Dtype::DtypeCode::Float,     4, "Float32"  );
const Dtype Dtype::Float64   = Dtype(Dtype::DtypeCode::Float,     8, "Float64"  );
const Dtype Dtype::Int32     = Dtype(Dtype::DtypeCode::Int,       4, "Int32"    );
const Dtype Dtype::Int64     = Dtype(Dtype::DtypeCode::Int,       8, "Int64"    );
const Dtype Dtype::UInt8     = Dtype(Dtype::DtypeCode::UInt,      1, "UInt8"    );
const Dtype Dtype::UInt16    = Dtype(Dtype::DtypeCode::UInt,      2, "UInt16"   );
const Dtype Dtype::Bool      = Dtype(Dtype::DtypeCode::Bool,      1, "Bool"     );
// clang-format on

}  // namespace core
}  // namespace open3d

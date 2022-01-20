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

#include "open3d/core/nns/blockselect/BlockSelectImpl.cuh"

namespace open3d {
namespace core {
BLOCK_SELECT_IMPL(double, int32_t, true, 1, 1);
BLOCK_SELECT_IMPL(double, int32_t, false, 1, 1);

BLOCK_SELECT_IMPL(double, int32_t, true, 32, 2);
BLOCK_SELECT_IMPL(double, int32_t, false, 32, 2);

BLOCK_SELECT_IMPL(double, int32_t, true, 64, 3);
BLOCK_SELECT_IMPL(double, int32_t, false, 64, 3);

BLOCK_SELECT_IMPL(double, int32_t, true, 128, 3);
BLOCK_SELECT_IMPL(double, int32_t, false, 128, 3);

BLOCK_SELECT_IMPL(double, int32_t, true, 256, 4);
BLOCK_SELECT_IMPL(double, int32_t, false, 256, 4);

BLOCK_SELECT_IMPL(double, int32_t, true, 512, 8);
BLOCK_SELECT_IMPL(double, int32_t, false, 512, 8);

BLOCK_SELECT_IMPL(double, int32_t, true, 1024, 8);
BLOCK_SELECT_IMPL(double, int32_t, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(double, int32_t, true, 2048, 8);
BLOCK_SELECT_IMPL(double, int32_t, false, 2048, 8);
#endif

BLOCK_SELECT_IMPL(double, int64_t, true, 1, 1);
BLOCK_SELECT_IMPL(double, int64_t, false, 1, 1);

BLOCK_SELECT_IMPL(double, int64_t, true, 32, 2);
BLOCK_SELECT_IMPL(double, int64_t, false, 32, 2);

BLOCK_SELECT_IMPL(double, int64_t, true, 64, 3);
BLOCK_SELECT_IMPL(double, int64_t, false, 64, 3);

BLOCK_SELECT_IMPL(double, int64_t, true, 128, 3);
BLOCK_SELECT_IMPL(double, int64_t, false, 128, 3);

BLOCK_SELECT_IMPL(double, int64_t, true, 256, 4);
BLOCK_SELECT_IMPL(double, int64_t, false, 256, 4);

BLOCK_SELECT_IMPL(double, int64_t, true, 512, 8);
BLOCK_SELECT_IMPL(double, int64_t, false, 512, 8);

BLOCK_SELECT_IMPL(double, int64_t, true, 1024, 8);
BLOCK_SELECT_IMPL(double, int64_t, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(double, int64_t, true, 2048, 8);
BLOCK_SELECT_IMPL(double, int64_t, false, 2048, 8);
#endif

}  // namespace core
}  // namespace open3d
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

#pragma once

#include "Open3D/Container/Dispatch.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

/// \brief Returns true if two shapes are compatible for broadcasting.
/// \param l_shape Shape of the left-hand-side Tensor.
/// \param r_shape Shape of the left-hand-side Tensor.
/// \return Returns true if \p l_shape and \p r_shape are compatible for
/// broadcasting.
bool IsCompatibleBroadcastShape(const SizeVector& l_shape,
                                const SizeVector& r_shape);

/// \brief Returns the broadcasted shape of two shapes.
/// \param l_shape Shape of the left-hand-side Tensor.
/// \param r_shape Shape of the left-hand-side Tensor.
/// \return The broadcasted shape.
SizeVector BroadcastedShape(const SizeVector& l_shape,
                            const SizeVector& r_shape);

/// \brief Returns true if \p src_shape can be brocasted to \p dst_shape.
/// \param src_shape Source tensor shape.
/// \param dst_shape Destination tensor shape.
/// \return Returns true if \p src_shape can be brocasted to \p dst_shape.
bool CanBeBrocastedToShape(const SizeVector& src_shape,
                           const SizeVector& dst_shape);

}  // namespace open3d

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

#include "Open3D/Core/TensorKey.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

NoneType None;

TensorKey TensorKey::Index(int64_t index) {
    return TensorKey(TensorKeyMode::Index, index, 0, 0, 0, false, false, false,
                     Tensor());
}

TensorKey TensorKey::Slice(int64_t start, int64_t stop, int64_t step) {
    return Slice(start, stop, step, false, false, false);
}

TensorKey TensorKey::Slice(int64_t start, int64_t stop, NoneType step) {
    return Slice(start, stop, 0, false, false, true);
}

TensorKey TensorKey::Slice(int64_t start, NoneType stop, int64_t step) {
    return Slice(start, 0, step, false, true, false);
}

TensorKey TensorKey::Slice(int64_t start, NoneType stop, NoneType step) {
    return Slice(start, 0, 0, false, true, true);
}

TensorKey TensorKey::Slice(NoneType start, int64_t stop, int64_t step) {
    return Slice(0, stop, step, true, false, false);
}

TensorKey TensorKey::Slice(NoneType start, int64_t stop, NoneType step) {
    return Slice(0, stop, 0, true, false, true);
}

TensorKey TensorKey::Slice(NoneType start, NoneType stop, int64_t step) {
    return Slice(0, 0, step, true, true, false);
}

TensorKey TensorKey::Slice(NoneType start, NoneType stop, NoneType step) {
    return Slice(0, 0, 0, true, true, true);
}

TensorKey TensorKey::IndexTensor(const Tensor& index_tensor) {
    return TensorKey(TensorKeyMode::IndexTensor, 0, 0, 0, 0, false, false,
                     false, index_tensor);
}

std::shared_ptr<Tensor> TensorKey::GetIndexTensor() const {
    AssertMode(TensorKeyMode::IndexTensor);
    return index_tensor_;
}

TensorKey TensorKey::UpdateWithDimSize(int64_t dim_size) const {
    AssertMode(TensorKeyMode::Slice);
    return TensorKey(TensorKeyMode::Slice, 0, start_is_none_ ? 0 : start_,
                     stop_is_none_ ? dim_size : stop_,
                     step_is_none_ ? 1 : step_, false, false, false, Tensor());
}

TensorKey TensorKey::Slice(int64_t start,
                           int64_t stop,
                           int64_t step,
                           bool start_is_none,
                           bool stop_is_none,
                           bool step_is_none) {
    return TensorKey(TensorKeyMode::Slice, 0, start, stop, step, start_is_none,
                     stop_is_none, step_is_none, Tensor());
}

TensorKey::TensorKey(TensorKeyMode mode,
                     int64_t index,
                     int64_t start,
                     int64_t stop,
                     int64_t step,
                     bool start_is_none,
                     bool stop_is_none,
                     bool step_is_none,
                     const Tensor& index_tensor)
    : mode_(mode),
      index_(index),
      start_(start),
      stop_(stop),
      step_(step),
      start_is_none_(start_is_none),
      stop_is_none_(stop_is_none),
      step_is_none_(step_is_none),
      index_tensor_(std::make_shared<Tensor>(index_tensor)){};

};  // namespace open3d

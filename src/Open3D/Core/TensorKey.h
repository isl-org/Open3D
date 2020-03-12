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

#include "Open3D/Utility/Console.h"

namespace open3d {

class NoneType {};

extern NoneType None;

/// A class to represent one of:
/// 1) tensor index
/// e.g. t[0], t[2]
/// 2) tensor slice
/// e.g. t[0:10:2], t[:-1], t[3:]
///
/// Example usage:
/// ```cpp
/// Tensor x({2, 3, 4}, Dtype::Float32);
/// // Equivalent to y = x[1, :3, 0:-1:2] in Python
/// Tensor y = t.GetItem({TensorKey::Index(1),
///                       TensorKey::Slice(None, 3, None),
///                       TensorKey::Slice(0, -1, 2)});
/// ```
class TensorKey {
public:
    enum class TensorKeyMode { Index, Slice };

    /// Construct an indexing TensorKey
    static TensorKey Index(int64_t index) {
        return TensorKey(TensorKeyMode::Index, index, 0, 0, 0, false, false,
                         false);
    }

    /// Construct a slicing TensorKey
    static TensorKey Slice(int64_t start, int64_t stop, int64_t step) {
        return Slice(start, stop, step, false, false, false);
    }
    static TensorKey Slice(int64_t start, int64_t stop, NoneType step) {
        return Slice(start, stop, 0, false, false, true);
    }
    static TensorKey Slice(int64_t start, NoneType stop, int64_t step) {
        return Slice(start, 0, step, false, true, false);
    }
    static TensorKey Slice(int64_t start, NoneType stop, NoneType step) {
        return Slice(start, 0, 0, false, true, true);
    }
    static TensorKey Slice(NoneType start, int64_t stop, int64_t step) {
        return Slice(0, stop, step, true, false, false);
    }
    static TensorKey Slice(NoneType start, int64_t stop, NoneType step) {
        return Slice(0, stop, 0, true, false, true);
    }
    static TensorKey Slice(NoneType start, NoneType stop, int64_t step) {
        return Slice(0, 0, step, true, true, false);
    }
    static TensorKey Slice(NoneType start, NoneType stop, NoneType step) {
        return Slice(0, 0, 0, true, true, true);
    }

    /// Getters will check the TensorKeyMode
    TensorKeyMode GetMode() const { return mode_; }

    int64_t GetIndex() const {
        AssertMode(TensorKeyMode::Index);
        return index_;
    }
    int64_t GetStart() const {
        AssertMode(TensorKeyMode::Slice);
        return start_;
    }
    int64_t GetStop() const {
        AssertMode(TensorKeyMode::Slice);
        return stop_;
    }
    int64_t GetStep() const {
        AssertMode(TensorKeyMode::Slice);
        return step_;
    }
    bool GetStartIsNone() const {
        AssertMode(TensorKeyMode::Slice);
        return start_is_none_;
    }
    bool GetStopIsNone() const {
        AssertMode(TensorKeyMode::Slice);
        return stop_is_none_;
    }
    bool GetStepIsNone() const {
        AssertMode(TensorKeyMode::Slice);
        return step_is_none_;
    }

    /// When dim_size is know, convert the slice object such that
    /// start_is_none_ == stop_is_none_ == step_is_none_ == false
    /// E.g. if t.shape == (5,), t[:4]:
    ///      before compute: Slice(None,    4, None)
    ///      after compute : Slice(   0,    4,    1)
    /// E.g. if t.shape == (5,), t[1:]:
    ///      before compute: Slice(   1, None, None)
    ///      after compute : Slice(   1,    5,    1)
    TensorKey UpdateWithDimSize(int64_t dim_size) const {
        AssertMode(TensorKeyMode::Slice);
        return TensorKey(TensorKeyMode::Slice, 0, start_is_none_ ? 0 : start_,
                         stop_is_none_ ? dim_size : stop_,
                         step_is_none_ ? 1 : step_, false, false, false);
    }

protected:
    /// The fully specifiec slice factory shall not be called directly.
    static TensorKey Slice(int64_t start,
                           int64_t stop,
                           int64_t step,
                           bool start_is_none,
                           bool stop_is_none,
                           bool step_is_none) {
        return TensorKey(TensorKeyMode::Slice, 0, start, stop, step,
                         start_is_none, stop_is_none, step_is_none);
    }

    /// The fully specified constructor shall not be called directly. Use the
    /// factory functions instead.
    TensorKey(TensorKeyMode mode,
              int64_t index,
              int64_t start,
              int64_t stop,
              int64_t step,
              bool start_is_none,
              bool stop_is_none,
              bool step_is_none)
        : mode_(mode),
          index_(index),
          start_(start),
          stop_(stop),
          step_(step),
          start_is_none_(start_is_none),
          stop_is_none_(stop_is_none),
          step_is_none_(step_is_none){};

    void AssertMode(TensorKeyMode mode) const {
        if (mode != mode_) {
            utility::LogError("Wrong TensorKeyMode.");
        }
    }

    /// Depending on the mode, some properties may or may not be used.
    TensorKeyMode mode_;

public:
    /// Properties for TensorKeyMode::Index.
    int64_t index_ = 0;

    /// Properties for TensorKeyMode::Slice.
    int64_t start_ = 0;
    int64_t stop_ = 0;
    int64_t step_ = 0;
    bool start_is_none_ = false;
    bool stop_is_none_ = false;
    bool step_is_none_ = false;
};

};  // namespace open3d

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

class Tensor;  // Avoids circular include

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
    enum class TensorKeyMode { Index, Slice, IndexTensor };

    /// Construct an TensorKeyMode::Index type TensorKey.
    /// E.g. b = a[3]
    static TensorKey Index(int64_t index);

    /// Construct an TensorKeyMode::Slice type TensorKey.
    /// E.g. b = a[0:100:2]
    static TensorKey Slice(int64_t start, int64_t stop, int64_t step);
    static TensorKey Slice(int64_t start, int64_t stop, NoneType step);
    static TensorKey Slice(int64_t start, NoneType stop, int64_t step);
    static TensorKey Slice(int64_t start, NoneType stop, NoneType step);
    static TensorKey Slice(NoneType start, int64_t stop, int64_t step);
    static TensorKey Slice(NoneType start, int64_t stop, NoneType step);
    static TensorKey Slice(NoneType start, NoneType stop, int64_t step);
    static TensorKey Slice(NoneType start, NoneType stop, NoneType step);

    /// Construct an TensorKeyMode::IndexTensor type TensorKey (advnced
    /// indexing).
    static TensorKey IndexTensor(const Tensor& index_tensor);

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

    std::shared_ptr<Tensor> GetIndexTensor() const;

    /// When dim_size is know, convert the slice object such that
    /// start_is_none_ == stop_is_none_ == step_is_none_ == false
    /// E.g. if t.shape == (5,), t[:4]:
    ///      before compute: Slice(None,    4, None)
    ///      after compute : Slice(   0,    4,    1)
    /// E.g. if t.shape == (5,), t[1:]:
    ///      before compute: Slice(   1, None, None)
    ///      after compute : Slice(   1,    5,    1)
    TensorKey UpdateWithDimSize(int64_t dim_size) const;

protected:
    /// The fully specifiec slice factory shall not be called directly.
    static TensorKey Slice(int64_t start,
                           int64_t stop,
                           int64_t step,
                           bool start_is_none,
                           bool stop_is_none,
                           bool step_is_none);

    /// The fully specified constructor shall not be called directly. Use the
    /// factory functions instead.
    TensorKey(TensorKeyMode mode,
              int64_t index,
              int64_t start,
              int64_t stop,
              int64_t step,
              bool start_is_none,
              bool stop_is_none,
              bool step_is_none,
              const Tensor& index_tensor);

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

    /// Properties for TensorKeyMode::IndexTensor.
    /// To avoid circular include, the pointer type is used. The index_tensor is
    /// shallow-copied when the TensorKey constructor is called.
    std::shared_ptr<Tensor> index_tensor_;
};

};  // namespace open3d

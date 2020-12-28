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

#include "open3d/utility/Console.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

// Avoids circular include.
class Tensor;

// Same as utility::nullopt. Provides a similar Python slicing API.
constexpr utility::nullopt_t None{utility::nullopt_t::init()};

/// \brief TensorKey is used to represent single index, slice or advanced
/// indexing on a Tensor.
///
/// See https://numpy.org/doc/stable/reference/arrays.indexing.html for details.
class TensorKey {
public:
    /// Instantiates a TensorKey with single index mode.
    ///
    /// t[0]   : t.GetItem({TensorKey::Index(0)})
    /// t[2]   : t.GetItem({TensorKey::Index(2)})
    /// t[2, 3]: t.GetItem({TensorKey::Index(2), TensorKey::Index(3)})
    ///
    /// \param index: Index to the tensor.
    static TensorKey Index(int64_t index);

    /// Instantiates a TensorKey with slice mode.
    ///
    /// t[0:10:2]: t.GetItem({TensorKey::Slice(0    , 10  , 2   )})
    /// t[:-1]   : t.Getitem({TensorKey::Slice(None , None, -1  )})
    /// t[3:]    : t.GetItem({TensorKey::Slice(3    , None, None)})
    ///
    /// \param start: Start index. None means starting from the 0-th element.
    /// \param stop: Stop index. None means stopping at the last element.
    /// \param step: Step size. None means step size 1.
    static TensorKey Slice(utility::optional<int64_t> start,
                           utility::optional<int64_t> stop,
                           utility::optional<int64_t> step);

    /// Instantiates a TensorKey with tensor-index (advanced indexing) mode.
    ///
    /// t[[1, 2], [3:]]: advanced indexing on dim-0, slicing on dim-1.
    /// t.GetItem({
    ///     TensorKey::IndexTensor(Tensor::Init<int64_t>({1, 2}, device),
    ///     TensorKey::Slice(3, None, None),
    /// });
    ///
    /// \param index_tensor: Indexing tensor of dtype int64_t or bool.
    static TensorKey IndexTensor(const Tensor& index_tensor);

    enum class TensorKeyMode { Index, Slice, IndexTensor };
    ~TensorKey() {}

    /// Returns TensorKey mode.
    TensorKeyMode GetMode() const;

    /// Convert TensorKey to a string representation.
    std::string ToString() const;

public:
    /// Get (single) index.
    /// For TensorKeyMode::Index only.
    int64_t GetIndex() const;

    /// Get start index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    int64_t GetStart() const;

    /// Get stop index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    int64_t GetStop() const;

    /// Get step index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    int64_t GetStep() const;

    /// When dim_size is know, conver the None values in start, stop, step with
    /// to concrete values and returns a new TensorKey.
    /// E.g. if t.shape == (5,), t[:4]:
    ///      before compute: Slice(None,    4, None)
    ///      after compute : Slice(   0,    4,    1)
    /// E.g. if t.shape == (5,), t[1:]:
    ///      before compute: Slice(   1, None, None)
    ///      after compute : Slice(   1,    5,    1)
    /// For TensorKeyMode::Slice only.
    TensorKey InstantiateDimSize(int64_t dim_size) const;

    /// Get advanced indexing tensor.
    /// For TensorKeyMode::IndexTensor only.
    Tensor GetIndexTensor() const;

private:
    class Impl;
    class IndexImpl;
    class SliceImpl;
    class IndexTensorImpl;
    std::shared_ptr<Impl> impl_;
    TensorKey(const std::shared_ptr<Impl>& impl);
};

}  // namespace core
}  // namespace open3d

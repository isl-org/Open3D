// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
//

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/custom_class.h"
#include "torch/script.h"

/// A RaggedTensor is a tensor with ragged dimensions, whose slice
/// may have different lengths. We define a container for ragged tensor
/// to support operations involving batches whose elements may have different
/// shape.
struct RaggedTensor : torch::CustomClassHolder {
public:
    RaggedTensor() {}

    /// Constructor for creating RaggedTensor with values and row_splits.
    RaggedTensor(torch::Tensor values, torch::Tensor row_splits)
        : _values(values), _row_splits(row_splits) {}

    /// Creates a RaggedTensor with rows partitioned by row_splits.
    ///
    /// The returned `RaggedTensor` corresponds with the python list defined by:
    /// ```python
    /// result = [values[row_splits[i]:row_splits[i + 1]]
    ///           for i in range(len(row_splits) - 1)]
    /// ```
    c10::intrusive_ptr<RaggedTensor> FromRowSplits(torch::Tensor values,
                                                   torch::Tensor row_splits);

    /// Returns _values tensor.
    torch::Tensor GetValues();

    /// Returns _row_splits tensor.
    torch::Tensor GetRowSplits();
    std::string ToString();

    torch::Tensor GetItem(int key);

private:
    torch::Tensor _values, _row_splits;
};

static auto registry =
        torch::class_<RaggedTensor>("my_classes", "RaggedTensor")
                .def(torch::init<>())
                .def("from_row_splits", &RaggedTensor::FromRowSplits)
                .def("get_values", &RaggedTensor::GetValues)
                .def("get_row_splits", &RaggedTensor::GetRowSplits)
                .def("__repr__",
                     [](const c10::intrusive_ptr<RaggedTensor>& self) {
                         return self->ToString();
                     })
                .def("__str__",
                     [](const c10::intrusive_ptr<RaggedTensor>& self) {
                         return self->ToString();
                     })
                .def("__getitem__",
                     [](const c10::intrusive_ptr<RaggedTensor>& self,
                        int64_t key) { return self->GetItem(key); });

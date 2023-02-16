# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import torch
import numpy as np

__all__ = ['RaggedTensor']


class RaggedTensor:
    """RaggedTensor.

    A RaggedTensor is a tensor with ragged dimension, whose slice may
    have different lengths. We define a container for ragged tensor to
    support operations involving batches whose elements may have different
    shape.

    """

    def __init__(self, r_tensor, internal=False):
        """Creates a `RaggedTensor` with specified torch script object.

        This constructor is private -- please use one of the following ops
        to build `RaggedTensor`'s:

        * `ml3d.classes.RaggedTensor.from_row_splits`

        Raises:
            ValueError: If internal = False. This method is intended for internal use.

        """
        if not internal:
            raise ValueError(
                "RaggedTensor constructor is private, please use one of the factory method instead(e.g. RaggedTensor.from_row_splits())"
            )
        self.r_tensor = r_tensor

    @classmethod
    def from_row_splits(cls, values, row_splits, validate=True, copy=True):
        """Creates a RaggedTensor with rows partitioned by row_splits.

        The returned `RaggedTensor` corresponds with the python list defined by::

            result = [values[row_splits[i]:row_splits[i + 1]]
                    for i in range(len(row_splits) - 1)]

        Args:
            values: A Tensor with shape [N, None].
            row_splits: A 1-D integer tensor with shape `[N+1]`. Must not be
                empty, and must be stored in ascending order. `row_splits[0]` must
                be zero and `row_splits[-1]` must be `N`.
            validate: Verify that `row_splits` are compatible with `values`.
                Set it to False to avoid expensive checks.
            copy: Whether to do a deep copy for `values` and `row_splits`.
                Set it to False to save memory for short term usage.

        Returns:
            A `RaggedTensor` container.

        Example:

        >>> print(ml3d.classes.RaggedTensor.from_row_splits(
        ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
        ...     row_splits=[0, 4, 4, 7, 8, 8]))
        <RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

        """
        if isinstance(values, list):
            values = torch.tensor(values, dtype=torch.float64)
        elif isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        elif isinstance(values, torch.Tensor) and copy:
            values = values.clone()

        if isinstance(row_splits, list):
            row_splits = torch.tensor(row_splits, dtype=torch.int64)
        elif isinstance(row_splits, np.ndarray):
            row_splits = torch.from_numpy(row_splits)
        elif isinstance(row_splits, torch.Tensor) and copy:
            row_splits = row_splits.clone()

        r_tensor = torch.classes.my_classes.RaggedTensor().from_row_splits(
            values, row_splits, validate)
        return cls(r_tensor, internal=True)

    @property
    def values(self):
        """The concatenated rows for this ragged tensor."""
        return self.r_tensor.get_values()

    @property
    def row_splits(self):
        """The row-split indices for this ragged tensor's `values`."""
        return self.r_tensor.get_row_splits()

    @property
    def dtype(self):
        """The `DType` of values in this ragged tensor."""
        return self.values.dtype

    @property
    def device(self):
        """The device of values in this ragged tensor."""
        return self.values.device

    @property
    def shape(self):
        """The statically known shape of this ragged tensor."""
        return [len(self.r_tensor), None, *self.values.shape[1:]]

    @property
    def requires_grad(self):
        """Read/writeble `requires_grad` for values."""
        return self.values.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.values.requires_grad = value

    def clone(self):
        """Returns a clone of object."""
        return RaggedTensor(self.r_tensor.clone(), True)

    def to_list(self):
        """Returns a list of tensors"""
        return [tensor for tensor in self.r_tensor]

    def __getitem__(self, idx):
        return self.r_tensor[idx]

    def __repr__(self):
        return f"RaggedTensor(values={self.values}, row_splits={self.row_splits})"

    def __len__(self):
        return len(self.r_tensor)

    def __add__(self, other):
        return RaggedTensor(self.r_tensor + self.__convert_to_tensor(other),
                            True)

    def __iadd__(self, other):
        self.r_tensor += self.__convert_to_tensor(other)
        return self

    def __sub__(self, other):
        return RaggedTensor(self.r_tensor - self.__convert_to_tensor(other),
                            True)

    def __isub__(self, other):
        self.r_tensor -= self.__convert_to_tensor(other)
        return self

    def __mul__(self, other):
        return RaggedTensor(self.r_tensor * self.__convert_to_tensor(other),
                            True)

    def __imul__(self, other):
        self.r_tensor *= self.__convert_to_tensor(other)
        return self

    def __truediv__(self, other):
        return RaggedTensor(self.r_tensor / self.__convert_to_tensor(other),
                            True)

    def __itruediv__(self, other):
        self.r_tensor /= self.__convert_to_tensor(other)
        return self

    def __floordiv__(self, other):
        return RaggedTensor(self.r_tensor // self.__convert_to_tensor(other),
                            True)

    def __ifloordiv__(self, other):
        self.r_tensor //= self.__convert_to_tensor(other)
        return self

    def __convert_to_tensor(self, value):
        """Converts scalar/tensor/RaggedTensor to torch.Tensor"""
        if isinstance(value, RaggedTensor):
            if self.row_splits.shape != value.row_splits.shape or torch.any(
                    self.row_splits != value.row_splits).item():
                raise ValueError(
                    f"Incompatible shape : {self.row_splits} and {value.row_splits}"
                )
            return value.values
        elif isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, (int, float, bool)):
            return torch.Tensor([value]).to(type(value))
        else:
            raise ValueError(f"Unknown type : {type(value)}")

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import paddle
import numpy as np

__all__ = ['RaggedTensor']


class RaggedTensor:

    def __init__(self, values, row_splits, internal=False):
        if not internal:
            raise ValueError(
                "RaggedTensor constructor is private, please use one of the factory method instead(e.g. RaggedTensor.from_row_splits())"
            )
        self._values = values
        self._row_splits = row_splits

    @classmethod
    def _from_row_splits(cls, values, row_splits, validate=True):
        if row_splits.dtype != paddle.int64:
            raise ValueError("row_splits must have type paddle.int64")

        values = values.contiguous()
        row_splits = row_splits.contiguous()

        if validate:
            if len(row_splits.shape) != 1:
                raise ValueError("row_splits must be of rank 1")
            if row_splits[0] != 0:
                raise ValueError(
                    f"Arguments to from_row_splits do not form a valid RaggedTensor. Expect row_splits[0] == 0 but received row_splits[0] == {row_splits[0]}."
                )
            for i in range(0, row_splits.shape[0] - 1):
                if row_splits[i] > row_splits[i + 1]:
                    raise ValueError(
                        "row_splits must be monotonically increasing")

        row_splits = row_splits.to(values.place)

        return values, row_splits

    @classmethod
    def from_row_splits(cls, values, row_splits, validate=True, copy=True):

        if isinstance(values, list):
            values = paddle.to_tensor(values, dtype=paddle.float64)
        elif isinstance(values, np.ndarray):
            values = paddle.to_tensor(values)
        elif isinstance(values, paddle.Tensor) and copy:
            values = values.clone()

        if isinstance(row_splits, list):
            row_splits = paddle.to_tensor(row_splits, dtype=paddle.int64)
        elif isinstance(row_splits, np.ndarray):
            row_splits = paddle.to_tensor(row_splits)
        elif isinstance(row_splits, paddle.Tensor) and copy:
            row_splits = row_splits.clone()

        values, row_splits = cls._from_row_splits(values, row_splits, validate)

        return cls(values, row_splits, internal=True)

    @property
    def values(self):
        """The concatenated rows for this ragged tensor."""
        return self._values

    @property
    def row_splits(self):
        """The row-split indices for this ragged tensor's `values`."""
        return self._row_splits

    @property
    def dtype(self):
        """The `DType` of values in this ragged tensor."""
        return self._values.dtype

    @property
    def device(self):
        """The device of values in this ragged tensor."""
        return self._values.place

    @property
    def shape(self):
        """The statically known shape of this ragged tensor."""
        return [
            len(self._row_splits.shape[0] - 1), None, *self._values.shape[1:]
        ]

    @property
    def requires_grad(self):
        """Read/writeble `requires_grad` for values."""
        return not self._values.stop_gradient

    @requires_grad.setter
    def requires_grad(self, value):
        # NOTE: stop_gradient=True means not requires grad
        self._values.stop_gradient = not value

    def clone(self):
        """Returns a clone of object."""
        return self.__class__(self._values.clone(), self._row_splits.clone(),
                              True)

    def to_list(self):
        """Returns a list of tensors"""
        return [tensor for tensor in self._values]

    def __getitem__(self, idx):
        return self._values.slice([
            0,
        ], [
            self._row_splits[idx],
        ], [
            self._row_splits[idx + 1],
        ])

    def __repr__(self):
        return f"RaggedTensor(values={self._values}, row_splits={self._row_splits})"

    def __len__(self):
        return len(self._row_splits.shape[0] - 1)

    def __add__(self, other):
        values, row_splits = self.__class__._from_row_splits(
            self._values + self.__convert_to_tensor(other), self._row_splits,
            False)
        return RaggedTensor(values, row_splits, True)

    def __iadd__(self, other):
        paddle.assign(self._values + self.__convert_to_tensor(other),
                      self._values)
        return self

    def __sub__(self, other):
        values, row_splits = self.__class__._from_row_splits(
            self._values - self.__convert_to_tensor(other), self._row_splits,
            False)
        return RaggedTensor(values.clone(), row_splits.clone(), True)

    def __isub__(self, other):
        paddle.assign(self._values - self.__convert_to_tensor(other),
                      self._values)
        return self

    def __mul__(self, other):
        values, row_splits = self.__class__._from_row_splits(
            self._values * self.__convert_to_tensor(other), self._row_splits,
            False)
        return RaggedTensor(values.clone(), row_splits.clone(), True)

    def __imul__(self, other):
        paddle.assign(self._values * self.__convert_to_tensor(other),
                      self._values)
        return self

    def __truediv__(self, other):
        values, row_splits = self.__class__._from_row_splits(
            self._values / self.__convert_to_tensor(other), self._row_splits,
            False)
        return RaggedTensor(values.clone(), row_splits.clone(), True)

    def __itruediv__(self, other):
        paddle.assign(self._values / self.__convert_to_tensor(other),
                      self._values)
        return self

    def __floordiv__(self, other):
        values, row_splits = self.__class__._from_row_splits(
            self._values // self.__convert_to_tensor(other), self._row_splits,
            False)
        return RaggedTensor(values.clone(), row_splits.clone(), True)

    def __ifloordiv__(self, other):
        paddle.assign(self._values // self.__convert_to_tensor(other),
                      self._values)
        return self

    def __convert_to_tensor(self, value):
        """Converts scalar/tensor/RaggedTensor to paddle.Tensor"""
        if isinstance(value, RaggedTensor):
            if self._row_splits.shape != value.row_splits.shape or paddle.any(
                    self._row_splits != value.row_splits).item():
                raise ValueError(
                    f"Incompatible shape : {self._row_splits} and {value.row_splits}"
                )
            return value.values
        elif isinstance(value, paddle.Tensor):
            return value
        elif isinstance(value, (int, float, bool)):
            return paddle.to_tensor([value], dtype=type(value))
        else:
            raise ValueError(f"Unknown type : {type(value)}")

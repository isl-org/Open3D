# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Paddle specific machine learning classes."""
import paddle

from .ragged_tensor import RaggedTensor

DTYPE_MAP = {
    paddle.bool: 'bool',
    paddle.float16: 'float16',
    paddle.float32: 'float32',
    paddle.float64: 'float64',
    paddle.int8: 'int8',
    paddle.int16: 'int16',
    paddle.int32: 'int32',
    paddle.int64: 'int64',
    paddle.bfloat16: 'uint16',
}

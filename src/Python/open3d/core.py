import open3d as o3d
import open3d.open3d_pybind as open3d_pybind
import numpy as np


def _numpy_dtype_to_dtype(numpy_dtype):
    if numpy_dtype == np.float32:
        return o3d.Dtype.Float32
    elif numpy_dtype == np.float64:
        return o3d.Dtype.Float64
    elif numpy_dtype == np.int32:
        return o3d.Dtype.Int32
    elif numpy_dtype == np.int64:
        return o3d.Dtype.Int64
    elif numpy_dtype == np.uint8:
        return o3d.Dtype.UInt8
    else:
        raise ValueError("Unsupported numpy dtype:", numpy_dtype)


class SizeVector(open3d_pybind.SizeVector):

    def __init__(self, values=None):
        if values is None:
            values = []
        # TODO: determine whether conversion can be done in C++ as well.
        if isinstance(values, tuple) or isinstance(values, list):
            values = np.array(values)
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError(
                "SizeVector only takes 1-D list, tuple or Numpy array.")
        super(SizeVector, self).__init__(values.astype(np.int64))


class Tensor(open3d_pybind.Tensor):
    """
    Open3D Tensor class. A Tensor is a view of data blob with shape, strides
    and etc. Tensor can be used to perform numerical operations.
    """

    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, tuple) or isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a list, tuple or Numpy array.")
        if dtype is None:
            dtype = _numpy_dtype_to_dtype(data.dtype)
        if device is None:
            device = o3d.Device("CPU:0")
        super(Tensor, self).__init__(data, dtype, device)

    def cuda(self, device_id=0):
        """
        Returns a copy of this tensor in CUDA memory.

        Args:
            device_id: CUDA device id.
        """
        return super(Tensor, self).cuda(device_id)

    def cpu(self):
        """
        Returns a copy of this tensor in CPU.

        If the Tensor is already in CPU, then no copy is performed.
        """
        return super(Tensor, self).cpu()

    def numpy(self):
        """
        Returns this tensor as a NumPy array. This tensor must be a CPU tensor,
        and the returned NumPy array shares the same memory as this tensor.
        Changes to the NumPy array will be reflected in the original tensor and
        vice versa.
        """
        return super(Tensor, self).numpy()

    @staticmethod
    def from_numpy(np_array):
        """
        Returns a Tensor from NumPy array. The resulting tensor is a CPU tensor
        that shares the same memory as the NumPy array. Changes to the tensor
        will be reflected in the original NumPy array and vice versa.

        Args:
            np_array: The Numpy array to be converted from.
        """
        return super(Tensor, Tensor).from_numpy(np_array)

    def to_dlpack(self):
        """
        Returns a DLPack PyCapsule representing this tensor.
        """
        return super(Tensor, self).to_dlpack()

    @staticmethod
    def from_dlpack(dlpack):
        """
        Returns a tensor converted from DLPack PyCapsule.
        """
        return super(Tensor, Tensor).from_dlpack(dlpack)

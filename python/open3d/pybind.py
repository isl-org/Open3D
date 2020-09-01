try:
    from open3d.cuda.pybind.core import cuda
    if cuda.is_available():
        from open3d.cuda.pybind import *
    else:
        raise ImportError("CUDA support not available.")
except ImportError:
    from open3d.cpu.pybind import *

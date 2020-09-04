try:
    from open3d.cuda.pybind.core import cuda as _cuda
    if _cuda.is_available():
        from open3d.cuda.pybind.ml.contrib import *
    else:
        raise ImportError("CUDA support not available.")
except ImportError:
    from open3d.cpu.pybind.ml.contrib import *

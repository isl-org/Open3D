# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open3D is a modern library for 3D data processing with both C++ and Python APIs. It includes:
- Core 3D data structures and algorithms
- GPU acceleration (CUDA, SYCL, ISPC)
- Machine learning support (PyTorch, TensorFlow)
- Visualization and rendering (GUI, WebRTC)
- Extensive pipeline support (registration, reconstruction, odometry)

## Build System

Open3D uses CMake with many configuration options. Key build commands:

```bash
# Standard build
mkdir build && cd build
cmake ..
make -j$(nproc)  # Linux/macOS
# or
cmake --build . --parallel  # Cross-platform

# Install dependencies (Ubuntu)
util/install_deps_ubuntu.sh

# Common CMake options
cmake -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_GUI=ON \
      -DBUILD_WEBRTC=ON \
      ..
```

Python package build:
```bash
cd python
pip install -e .  # Development install
# or
python setup.py develop
```

## Testing

**C++ Tests** (Google Test):
```bash
# Build with tests enabled
cmake -DBUILD_UNIT_TESTS=ON ..
make -j$(nproc)

# Run all C++ tests
./bin/tests

# Run specific test
./bin/tests --gtest_filter="*PointCloud*"
```

**Python Tests** (pytest):
```bash
cd python
pytest test/
# or specific test
pytest test/core/test_tensor_function.py::test_specific_function
```

## Architecture

### Core Structure
- `cpp/open3d/core/`: Core tensor operations, memory management, device abstraction
- `cpp/open3d/geometry/`: 3D geometry classes (PointCloud, TriangleMesh, Image, etc.)
- `cpp/open3d/pipelines/`: High-level algorithms (registration, reconstruction, odometry)
- `cpp/open3d/t/`: New tensor-based geometry API (more GPU-friendly)
- `cpp/open3d/visualization/`: Rendering and GUI systems
- `cpp/open3d/io/`: File I/O for various formats
- `cpp/open3d/ml/`: Machine learning operations

### Python Structure
- `python/open3d/`: Python API mirrors C++ structure
- `python/test/`: Python unit tests
- `cpp/pybind/`: pybind11 bindings connecting C++ to Python

### Device Support
- CPU: Default backend with OpenMP parallelization
- CUDA: GPU acceleration for core operations (when BUILD_CUDA_MODULE=ON)
- SYCL: Intel GPU support (when BUILD_SYCL_MODULE=ON)
- ISPC: SIMD acceleration (when BUILD_ISPC_MODULE=ON)

### Key Patterns
1. **Legacy vs Tensor API**: Old geometry classes vs new `t::geometry` tensor-based classes
2. **Device abstraction**: Code supports CPU/CUDA/SYCL through Device and MemoryManager
3. **Pybind integration**: C++ classes exposed to Python with automatic memory management
4. **Pipeline architecture**: Modular algorithms that can be chained together

## Development Workflow

1. **Build configuration**: Use cmake options to enable needed features
2. **Code style**: Follow existing patterns, especially device-agnostic code
3. **Testing**: Add both C++ (gtest) and Python (pytest) tests for new features
4. **Memory management**: Be aware of device memory when working with GPU code
5. **Documentation**: Update relevant .rst files in docs/ when adding public APIs

## Common Issues

- **CUDA**: Requires compatible GPU and CUDA toolkit version
- **Dependencies**: Many optional dependencies (see 3rdparty/)
- **Memory**: Large 3D datasets can exhaust memory, consider chunking
- **Threading**: OpenMP used throughout, be careful with thread safety
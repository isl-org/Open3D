## Master

* Python 3.9 support. Tensorflow bump 2.4.1 -> 2.5.0. PyTorch bump 1.7.1 -> 1.8.1 (LTS)
* Fix undefined names: docstr and VisibleDeprecationWarning (PR #3844)
* Corrected documentation for Tensor based PointClound, LineSet, TriangleMesh (PR #4685)

## 0.13

* CUDA support 10.1 -> 11.0. Tensorflow 2.3.1 -> 2.4.1. PyTorch 1.6.0 -> 1.7.1 (PR #3049). This requires a custom PyTorch wheel from https://github.com/isl-org/open3d_downloads/releases/tag/torch1.7.1 due to PyTorch issue #52663

## 0.12

* RealSense SDK v2 integrated for reading RS bag files (PR #2646)
* Tensor based RGBDImage class, Python bindings for Image and RGBDImage
* RealSense sensor configuration, live capture and recording (with example and tutorial) (PR #2748)
* Add mouselook for the legacy visualizer (PR #2551)
* Add function to randomly downsample pointcloud (PR #3050)
* Allow TriangleMesh with textures to be added (PR #3170)
* Python property of open3d.visualization.rendering.Open3DScene `get_view` has been renamed to `view`.
* Added LineSet::CreateCameraVisualization() for creating a simple camera visualization from intrinsic and extrinsic matrices (PR #3255)

## 0.11

* Fixes bug for preloading libc++ and libc++abi in Python
* Added GUI widgets and model-viewing app
* Fixes travis for race-condition on macOS
* Fixes appveyor configuration and to build all branches
* Updated travis.yml to support Ubuntu 18.04, gcc-7, and clang-7.0
* Contributors guidelines updated
* Avoid cstdlib random generators in ransac registration, use C++11 random instead.
* Fixed a bug in open3d::geometry::TriangleMesh::ClusterConnectedTriangles.
* Added option BUILD_BENCHMARKS for building microbenchmarks
* Extend Python API of UniformTSDFVolume to allow setting the origin
* Corrected documentation of PointCloud.h
* Added ISS Keypoint Detector
* Added an RPC interface for external visualizers running in a separate process
* Added `maximum_error` and `boundary_weight` parameter to `simplify_quadric_decimation`
* Remove support for Python 3.5
* Development wheels are available for user testing. See [Getting Started](http://www.open3d.org/docs/latest/getting_started.html) page for installation.
* PointCloud File IO support for new tensor data types.
* New PointCloud format support: XYZI (ASCII).
* Fast compression mode for PNG writing. (Issue #846)
* Ubuntu 20.04 (Focal) support.
* Added Line3D/Ray3D/Segment3D classes with plane, point, closest-distance, and AABB tests
* Updated Open3D.h.in to add certain missing header files
* Add Open3D-ML to Open3D wheel
* Fix a bug in PointCloud file format, use `float` instead of `float_t`
* Add per-point covariance member for geometry::PointCloud class.
* Add Generalized ICP implementation.

## 0.9.0

* Version bump to 0.9.0

## Master

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
* Add Open3D-ML to Open3D wheel
* Fix a bug in PointCloud file format, use `float` instead of `float_t`

## 0.9.0

* Version bump to 0.9.0

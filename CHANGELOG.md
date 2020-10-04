## Master

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
* Added `maximum_error` parameter to `simplify_quadric_decimation`

## 0.9.0

* Version bump to 0.9.0

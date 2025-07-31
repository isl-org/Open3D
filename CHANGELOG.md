## Main
-   Corrected documentation for Link Open3D in C++ projects (broken links).
-   Fix DLLs not being found in Python-package. Also prevent PATH from being searched for DLLs, except CUDA (PR #7108)
-   Fix MSAA sample count not being copied when FilamentView is copied
-   Fix TriangleMesh::SamplePointsUniformly and TriangleMesh::SamplePointsPoissonDisk now sampling colors from mesh if available (PR #6842)
-   Fix TriangleMesh::SamplePointsUniformly not sampling triangle meshes uniformly (PR #6653)
-   Fix tensor based TSDF integration example.
-   Use GLIBCXX_USE_CXX11_ABI=ON by default
-   Python 3.9 support. Tensorflow bump 2.4.1 -> 2.5.0. PyTorch bump 1.7.1 -> 1.8.1 (LTS)
-   Fix undefined names: docstr and VisibleDeprecationWarning (PR #3844)
-   Corrected documentation for Tensor based PointClound, LineSet, TriangleMesh (PR #4685)
-   Corrected documentation for KDTree (typo in Notebook) (PR #4744)
-   Corrected documentation for visualisation tutorial
-   Remove `setuptools` and `wheel` from requirements for end users (PR #5020)
-   Fix various typos (PR #5070)
-   Exposed more functionality in SLAM and odometry pipelines
-   Fix for depth estimation for VoxelBlockGrid
-   Reserve fragment buffer for VoxelBlockGrid operations
-   Fix raycasting scene: Allow setting of number of threads that are used for building a raycasting scene
-   Fix Python bindings for CUDA device synchronization, voxel grid saving (PR #5425)
-   Support msgpack versions without cmake
-   Changed TriangleMesh to store materials in a list so they can be accessed by the material index (PR #5938)
-   Support multi-threading in the RayCastingScene function to commit scene (PR #6051).
-   Fix some bad triangle generation in TriangleMesh::SimplifyQuadricDecimation
-   Fix printing of tensor in gpu and add validation check for bounds of axis-aligned bounding box (PR #6444)
-   Python 3.11 support. bump pybind11 v2.6.2 -> v2.11.1
-   Check for support of CUDA Memory Pools at runtime (#4679)
-   Fix `toString`, `CreateFromPoints` methods and improve docs in `AxisAlignedBoundingBox`. 🐛📝
-   Migrate Open3d documentation to furo theme ✨ (#6470)
-   Expose Near Clip + Far Clip parameters to setup_camera in OffscreenRenderer (#6520)
-   Add Doppler ICP in tensor registration pipeline (PR #5237)
-   Rename master branch to main.
-   Support in memory loading of XYZ files
-   Fix geometry picker Error when LineSet objects are presented (PR #6499)
-   Fix mis-configured application .desktop link for the Open3D viewer when installing to a custom path (PR #6599)
-   Fix regression in printing cuda tensor from PR #6444 🐛
-   Add Python pathlib support for file IO (PR #6619)
-   Fix log error message for `probability` argument validation in `PointCloud::SegmentPlane` (PR #6622)
-   Fix macOS arm64 builds, add CI runner for macOS arm64 (PR #6695)
-   Fix KDTreeFlann possibly using a dangling pointer instead of internal storage and simplified its members (PR #6734)
-   Fix RANSAC early stop if no inliers in a specific iteration (PR #6789)
-   Fix segmentation fault (infinite recursion) of DetectPlanarPatches if multiple points have same coordinates (PR #6794)
-   `TriangleMesh`'s `+=` operator appends UVs regardless of the presence of existing features (PR #6728)
-   Fix build with fmt v10.2.0 (#6783)
-   Fix segmentation fault (lambda reference capture) of VisualizerWithCustomAnimation::Play (PR #6804)
-   Python 3.12 support
-   Add O3DVisualizer API to enable collapse control of verts in the side panel (PR #6865)
-   Split pybind declarations/definitions to avoid C++ types in Python docs (PR #6869)
-   Fix minimal oriented bounding box of MeshBase derived classes and add new unit tests (PR #6898)
-   Fix projection of point cloud to Depth/RGBD image if no position attribute is provided (PR #6880)
-   Add choice of voxel pooling mode when creating VoxelGrid from PointCloud (PR #6937)
-   Support lowercase types when reading PCD files (PR #6930)
-   Fix visualization/draw ICP example and add warnings (PR #6933)
-   Unified cloud initializer pipeline for ICP (fixes segfault colored ICP) (PR #6942)
-   Fix tensor EstimatePointWiseNormalsWithFastEigen3x3 (PR #6980)
-   Fix alpha shape reconstruction if alpha too small for point scale (PR #6998)
-   Fix render to depth image on Apple Retina displays (PR #7001)
-   Fix infinite loop in segment_plane if num_points < ransac_n (PR #7032)
-   Add select_by_index method to Feature class (PR #7039)
-   Add optional indices arg for fast computation of a small subset of FPFH features (PR #7118).
-   Fix CMake configuration summary incorrectly reporting `no` for system BLAS. (PR #7230)
-   Add error handling for insufficient correspondences in AdvancedMatching (PR #7234)
-   Exposed `get_plotly_fig` and modified `draw_plotly` to return the `Figure` it creates. (PR #7258)
-   Fix build with librealsense v2.44.0 and upcoming VS 2022 17.13 (PR #7074)
-   Fix `deprecated-declarations` warnings when compiling code with C++20 standard (PR #7303)
-   macOS x86_64 not longer supported, only macOS arm64 is supported.
## 0.13

-   CUDA support 10.1 -> 11.0. Tensorflow 2.3.1 -> 2.4.1. PyTorch 1.6.0 -> 1.7.1 (PR #3049). This requires a custom PyTorch wheel from <https://github.com/isl-org/open3d_downloads/releases/tag/torch1.7.1> due to PyTorch issue #52663

## 0.12

-   RealSense SDK v2 integrated for reading RS bag files (PR #2646)
-   Tensor based RGBDImage class, Python bindings for Image and RGBDImage
-   RealSense sensor configuration, live capture and recording (with example and tutorial) (PR #2748)
-   Add mouselook for the legacy visualizer (PR #2551)
-   Add function to randomly downsample pointcloud (PR #3050)
-   Allow TriangleMesh with textures to be added (PR #3170)
-   Python property of open3d.visualization.rendering.Open3DScene `get_view` has been renamed to `view`.
-   Added LineSet::CreateCameraVisualization() for creating a simple camera visualization from intrinsic and extrinsic matrices (PR #3255)

## 0.11

-   Fixes bug for preloading libc++ and libc++abi in Python
-   Added GUI widgets and model-viewing app
-   Fixes travis for race-condition on macOS
-   Fixes appveyor configuration and to build all branches
-   Updated travis.yml to support Ubuntu 18.04, gcc-7, and clang-7.0
-   Contributors guidelines updated
-   Avoid cstdlib random generators in ransac registration, use C++11 random instead.
-   Fixed a bug in open3d::geometry::TriangleMesh::ClusterConnectedTriangles.
-   Added option BUILD_BENCHMARKS for building microbenchmarks
-   Extend Python API of UniformTSDFVolume to allow setting the origin
-   Corrected documentation of PointCloud.h
-   Added ISS Keypoint Detector
-   Added an RPC interface for external visualizers running in a separate process
-   Added `maximum_error` and `boundary_weight` parameter to `simplify_quadric_decimation`
-   Remove support for Python 3.5
-   Development wheels are available for user testing. See [Getting Started](https://www.open3d.org/docs/latest/getting_started.html) page for installation.
-   PointCloud File IO support for new tensor data types.
-   New PointCloud format support: XYZI (ASCII).
-   Fast compression mode for PNG writing. (Issue #846)
-   Ubuntu 20.04 (Focal) support.
-   Added Line3D/Ray3D/Segment3D classes with plane, point, closest-distance, and AABB tests
-   Updated Open3D.h.in to add certain missing header files
-   Add Open3D-ML to Open3D wheel
-   Fix a bug in PointCloud file format, use `float` instead of `float_t`
-   Add per-point covariance member for geometry::PointCloud class.
-   Add Generalized ICP implementation.

## 0.9.0

-   Version bump to 0.9.0

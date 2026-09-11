# Open3D C++ API Map

Open3D **v0.20.0**.

**This file does not list class members** — the installed headers do that, with
signatures and Doxygen comments, always matching the binary you link against:

```bash
O3D=/path/to/open3d-devel-.../include
grep -n "" "$O3D/open3d/t/geometry/PointCloud.h" | less   # or open it in your editor
grep -rn "VoxelDownSample" "$O3D/open3d/t/geometry/"
```

What follows is what grep **cannot** tell you: which namespace and header own a
task, which of many similarly-named methods is the right one, what is current
versus legacy, and which behaviors will surprise you.

## Namespace and Header Routing

| You want to… | Namespace | Header |
|---|---|---|
| Numeric arrays, devices, linear algebra | `open3d::core` | `open3d/core/Tensor.h`, `Device.h`, `Dtype.h` |
| Neighbor search | `open3d::core::nns` | `open3d/core/nns/NearestNeighborSearch.h` |
| Point clouds, meshes, images, ray casting, TSDF | `open3d::t::geometry` | `open3d/t/geometry/PointCloud.h`, `TriangleMesh.h`, `RaycastingScene.h`, `VoxelBlockGrid.h` |
| Read/write geometry and images | `open3d::t::io` | `open3d/t/io/PointCloudIO.h`, `TriangleMeshIO.h`, `ImageIO.h` |
| ICP and registration | `open3d::t::pipelines::registration` | `.../registration/Registration.h`, `Feature.h`, `TransformationEstimation.h`, `RobustKernel.h` |
| RGB-D odometry | `open3d::t::pipelines::odometry` | `.../odometry/RGBDOdometry.h` |
| Dense SLAM | `open3d::t::pipelines::slam` | `.../slam/Model.h`, `Frame.h` |
| Fragment pose refinement | `open3d::t::pipelines::slac` | `.../slac/SLACOptimizer.h`, `ControlGrid.h` |
| Show something on screen | `open3d::visualization` | `open3d/visualization/utility/Draw.h` |
| Build a custom 3D app | `open3d::visualization::gui` / `::rendering` | `.../gui/Application.h`, `Window.h`, `SceneWidget.h`; `.../rendering/Open3DScene.h`, `MaterialRecord.h` |
| Sample data | `open3d::data` | `open3d/data/Dataset.h` |
| Logging | `open3d::utility` | `open3d/utility/Logging.h` |
| Legacy Eigen equivalents (fallback only) | `open3d::geometry`, `::io`, `::pipelines`, `::camera` | `open3d/geometry/PointCloud.h`, etc. |

Umbrella header: `open3d/Open3D.h` — includes everything public.

## Task → API

### Point clouds

| Task | Call |
|---|---|
| Load / save | `t::io::ReadPointCloud`, `WritePointCloud` |
| Reduce point count | `VoxelDownSample` (uniform grid, usual choice), `UniformDownSample`, `RandomDownSample`, `FarthestPointDownSample` (best coverage, slowest) |
| Normals | `EstimateNormals`, then `OrientNormalsToAlignWithDirection` / `OrientNormalsTowardsCameraLocation` / `OrientNormalsConsistentTangentPlane` |
| Denoise | `RemoveStatisticalOutliers` (density-adaptive), `RemoveRadiusOutliers`, `RemoveNonFinitePoints` |
| Take a subset | `SelectByIndex`, `SelectByMask` |
| Find planes / clusters | `SegmentPlane`, `ClusterDBSCAN` |
| Smooth | `SmoothMLS`, `SmoothLaplacian`, `SmoothTaubin` |
| Move / colorize | `Transform`, `Translate`, `Rotate`, `Scale`, `PaintUniformColor` |
| From RGB-D | `CreateFromDepthImage`, `CreateFromRGBDImage` (static) |
| Back to 2D | `ProjectToDepthImage`, `ProjectToRGBDImage` |
| Compare two clouds | `ComputeMetrics` (Chamfer, Hausdorff, F-score) |
| Access attributes | `GetPointPositions` / `SetPointPositions`, `GetPointAttr("…")` / `SetPointAttr` for custom ones |

### Meshes

| Task | Call |
|---|---|
| Load / save | `t::io::ReadTriangleMesh`, `WriteTriangleMesh` |
| Primitives | `CreateBox`, `CreateSphere`, `CreateCylinder`, `CreateCone`, `CreateTorus`, `CreateArrow`, `CreateCoordinateFrame`, `CreateText` (static) |
| Normals | `ComputeVertexNormals` (smooth shading), `ComputeTriangleNormals` (flat) |
| Reduce triangles | `SimplifyQuadricDecimation` |
| Repair | `FillHoles`, `RemoveNonManifoldEdges`, `RemoveUnreferencedVertices` |
| CSG | `BooleanUnion`, `BooleanIntersection`, `BooleanDifference` |
| Cut | `ClipPlane` (keep a half-space), `SlicePlane` (cross-section curves) |
| Mesh → points | `SamplePointsUniformly` |
| UVs and textures | `ComputeUVAtlas`, `BakeVertexAttrTextures`, `BakeTriangleAttrTextures`, `ProjectImagesToAlbedo` |
| Surface from a point cloud | legacy only — `geometry::TriangleMesh::CreateFromPointCloudPoisson` / `…AlphaShape` / `…BallPivoting` |

### Registration

| Task | Call |
|---|---|
| Refine a roughly-aligned pair | `MultiScaleICP` (preferred) or `ICP` |
| Pick the error metric | `TransformationEstimationPointToPlane` (needs normals, converges fastest), `…PointToPoint`, `…ForColoredICP`, `…Symmetric`, `…ForDopplerICP` |
| Downweight outliers | `RobustKernel` with a `RobustKernelMethod` — `L2Loss`, `L1Loss`, `HuberLoss`, `CauchyLoss`, `GMLoss`, `TukeyLoss`, `GeneralizedLoss` |
| Coarse / global alignment | `ComputeFPFHFeature` + `CorrespondencesFromFeatures` (`Feature.h`); full RANSAC/FGR is **legacy only** |
| Score an alignment | `EvaluateRegistration` |
| Covariance for a pose graph | `GetInformationMatrix` |
| Optimize many fragments jointly | legacy only — `pipelines::registration::PoseGraph` + `GlobalOptimization` |

### Reconstruction

| Task | Call |
|---|---|
| Fuse RGB-D frames | `t::geometry::VoxelBlockGrid` — `GetUniqueBlockCoordinates`, then `Integrate` |
| Get the surface out | `ExtractPointCloud`, `ExtractTriangleMesh` |
| Frame-to-frame camera pose | `t::pipelines::odometry::RGBDOdometryMultiScale` |
| Online dense SLAM | `t::pipelines::slam::Model` + `Frame` |
| Refine fragment poses | `t::pipelines::slac::RunSLACOptimizerForFragments` |

### Queries and search

| Task | Call |
|---|---|
| k nearest neighbors | `core::nns::NearestNeighborSearch` — `KnnIndex()` then `KnnSearch` |
| All neighbors within a radius | `FixedRadiusIndex(r)` then `FixedRadiusSearch` |
| A different radius per query | `MultiRadiusIndex` / `MultiRadiusSearch` |
| k neighbors capped by a radius | `HybridIndex` / `HybridSearch` |
| Ray/mesh intersection | `RaycastingScene::CastRays` |
| Distance to a surface | `ComputeDistance` (unsigned), `ComputeSignedDistance` (signed) |
| Inside/outside test | `ComputeOccupancy` |
| Closest point on a surface | `ComputeClosestPoints` |
| Visibility / counting hits | `TestOcclusions`, `CountIntersections`, `ListIntersections` |

`KDTreeFlann` is legacy — always use `core::nns` instead.

### Visualization

| Task | Call |
|---|---|
| Just look at it | `visualization::Draw({geometry})` |
| Several named, toggleable objects | `visualization::Draw({DrawObject(name, geom, visible), …})` |
| An image file, no window | `gui::Application::RenderToImage` with a `FilamentRenderer` + `Open3DScene` |
| Custom app with widgets | `gui::Application` + `gui::Window` + `gui::SceneWidget` + `rendering::Open3DScene` |
| Control appearance | `rendering::MaterialRecord` — set `.shader` to `"defaultLit"`, `"defaultUnlit"`, `"normals"`, `"depth"`, `"unlitLine"` |

## Tensor ↔ Legacy

Same class name, different namespace: `PointCloud`, `TriangleMesh`, `LineSet`,
`Image`, `RGBDImage`, `AxisAlignedBoundingBox`, `OrientedBoundingBox`, and the
`Read*` / `Write*` IO functions (`io::X` → `t::io::X`).

Renamed or restructured:

| Legacy | Tensor |
|---|---|
| `pipelines::registration::RegistrationICP` | `t::pipelines::registration::ICP` / `MultiScaleICP` |
| `pipelines::odometry::*` | `t::pipelines::odometry::*` |
| `pipelines::integration::TSDFVolume` | `t::geometry::VoxelBlockGrid` (block-sparse; not drop-in) |
| `geometry::KDTreeFlann` + `KDTreeSearchParam*` | `core::nns::NearestNeighborSearch` — **always migrate** |
| Eigen matrices in public APIs | `core::Tensor` |
| `visualization::DrawGeometries` / `Visualizer` | `visualization::Draw`, `gui::Application` + `rendering::Open3DScene`, `O3DVisualizer` |

Tensor geometry types provide static `FromLegacy(...)` and member `ToLegacy()`.

### Legacy-only — a fallback here is justified

`camera::*` · `geometry::Octree` · `geometry::TetraMesh` ·
`geometry::HalfEdgeTriangleMesh` · `pipelines::integration::*` ·
pose graphs and `GlobalOptimization` · RANSAC and FGR global registration ·
`pipelines::color_map::*` · surface reconstruction from point clouds ·
legacy OpenGL `Visualizer` subclasses

### Tensor-only — no legacy equivalent

`core::Tensor` and explicit device placement · `core::HashMap` ·
`core::nns::NearestNeighborSearch` · `RaycastingScene` · `VoxelBlockGrid` ·
custom attributes via `TensorMap` · SLAC and SLAM · Doppler ICP

## Non-Obvious Behavior

Things that cost debugging time and are not visible in a signature.

| API | What to know |
|---|---|
| naming | C++ is `PascalCase` where Python is `snake_case`; getters are explicit (`GetPointPositions()` vs. `pcd.point["positions"]`) |
| dtype constants | Capitalized in C++ (`core::Float32`); lowercase in Python |
| `Tensor` copies | Shallow by default — use `Clone()` for a deep copy, `Contiguous()` when a kernel needs contiguous memory |
| `FromLegacy` | Defaults to `Float32` even though legacy geometry stores `double` |
| `RemoveStatisticalOutliers` | Returns `std::tuple<PointCloud, core::Tensor>` (cloud + mask), not just a cloud |
| any registration result | `transformation_` is always `Float64` on `CPU:0`, whatever device the inputs were on |
| `ExtractPointCloud` / `ExtractTriangleMesh` | `weight_threshold = 3.0f` by default — a voxel needs ≥3 observations, so few-frame integrations come back empty |
| `RaycastingScene` | CPU and SYCL only, never CUDA; device is a constructor argument. (The class comment in the header still says CPU-only — the binding docs and implementation are correct.) |
| `ComputeSignedDistance` | The sign is only meaningful for watertight meshes |
| `FixedRadiusSearch` | Ragged output: values plus row-split offsets |
| `EstimateNormals` | Orientation is arbitrary; follow with an `OrientNormals*` call |
| index tensors | Must be `Int64` |
| `utility::LogError` | `[[noreturn]]`; throws `std::runtime_error`. It is the error mechanism, not a print |
| device mixing | All inputs to a tensor pipeline must share a device; `To(device)` each one |
| `CreateRaysPinhole` | Two overloads — intrinsic/extrinsic matrices, or fov/center/eye/up |
| `data::*` | Downloads on first use; requires network access |
| `gui` | `Application::GetInstance().Initialize()` before any window; main thread only |
| CUDA / SYCL | A property of the binary you downloaded — check `core::cuda::IsAvailable()` / `core::sycl::IsAvailable()` at runtime |

## Sample Data

`open3d::data` classes download on construction; read `GetPath()`, `GetPaths()`,
or type-specific accessors (`GetDepthPaths()`, `GetColorPaths()`).

| Need | Use |
|---|---|
| A point cloud | `PCDPointCloud`, `PLYPointCloud`, `EaglePointCloud` |
| A mesh | `BunnyMesh`, `ArmadilloMesh`, `KnotMesh` |
| Two clouds to register | `DemoICPPointClouds`, `DemoColoredICPPointClouds`, `DemoFeatureMatchingPointClouds` |
| An RGB-D sequence | `SampleRedwoodRGBDImages`, `SampleFountainRGBDImages`, `BedroomRGBDImages`, `LoungeRGBDImages` |
| A textured model | `FlightHelmetModel`, `DamagedHelmetModel`, `MonkeyModel`, `CrateModel` |
| A full indoor scene | `LivingRoomPointClouds`, `OfficePointClouds`, `RedwoodIndoorLivingRoom1/2` |

Full list: `grep -n "^class" "$O3D/open3d/data/Dataset.h"`

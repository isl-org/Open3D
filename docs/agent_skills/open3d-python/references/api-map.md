# Open3D Python API Map

Open3D **v0.20.0**.

**This file does not list class members** — `dir()` and `help()` already do that,
better, and always in sync with your installed build:

```bash
python -c "import open3d as o3d; print([n for n in dir(o3d.t.geometry.PointCloud) if not n.startswith('_')])"
python -c "import open3d as o3d; help(o3d.t.geometry.PointCloud.voxel_down_sample)"
```

What follows is what introspection **cannot** tell you: which namespace owns a
task, which of ~60 similarly-named methods is the right one, what is current
versus legacy, and which behaviors will surprise you.

## Namespace Routing

| You want to… | Namespace |
|---|---|
| Numeric arrays, devices, linear algebra | `open3d.core` |
| Neighbor search | `open3d.core.nns` |
| Point clouds, meshes, images, bounding volumes, ray casting, TSDF | `open3d.t.geometry` |
| Read/write geometry and images; RealSense | `open3d.t.io` |
| ICP, feature matching, registration | `open3d.t.pipelines.registration` |
| RGB-D odometry | `open3d.t.pipelines.odometry` |
| Dense SLAM | `open3d.t.pipelines.slam` |
| Non-rigid refinement of fragment poses | `open3d.t.pipelines.slac` |
| Show something on screen | `open3d.visualization` (`draw`, `O3DVisualizer`) |
| Build a custom 3D app | `open3d.visualization.gui` + `.rendering` |
| Sample data | `open3d.data` |
| Legacy Eigen equivalents (fallback only) | `open3d.geometry`, `.io`, `.pipelines`, `.camera` |

## Task → API

### Point clouds

| Task | Call |
|---|---|
| Load / save | `t.io.read_point_cloud`, `t.io.write_point_cloud` |
| Reduce point count | `voxel_down_sample` (uniform grid, usual choice), `uniform_down_sample` (every k-th), `random_down_sample`, `farthest_point_down_sample` (best coverage, slowest) |
| Normals | `estimate_normals`, then one of `orient_normals_to_align_with_direction` / `orient_normals_towards_camera_location` / `orient_normals_consistent_tangent_plane` |
| Denoise | `remove_statistical_outliers` (density-adaptive), `remove_radius_outliers` (fixed radius), `remove_non_finite_points` |
| Take a subset | `select_by_index`, `select_by_mask`, `crop` |
| Find planes / clusters | `segment_plane` (one plane, RANSAC), `cluster_dbscan` (arbitrary clusters) |
| Smooth | `smooth_mls`, `smooth_bilateral`, `smooth_laplacian`, `smooth_taubin` |
| Move / colorize | `transform`, `translate`, `rotate`, `scale`, `paint_uniform_color` |
| From RGB-D | `create_from_depth_image`, `create_from_rgbd_image` (static) |
| Back to 2D | `project_to_depth_image`, `project_to_rgbd_image` |
| Compare two clouds | `compute_metrics` (Chamfer, Hausdorff, F-score) |

### Meshes

| Task | Call |
|---|---|
| Load / save | `t.io.read_triangle_mesh`, `t.io.write_triangle_mesh` |
| Primitives | `create_box`, `create_sphere`, `create_cylinder`, `create_cone`, `create_torus`, `create_arrow`, `create_coordinate_frame`, `create_text` (static) |
| Normals | `compute_vertex_normals` (smooth shading), `compute_triangle_normals` (flat) |
| Reduce triangles | `simplify_quadric_decimation(target_reduction=…)` |
| Repair | `fill_holes`, `remove_non_manifold_edges`, `remove_unreferenced_vertices` |
| CSG | `boolean_union`, `boolean_intersection`, `boolean_difference` |
| Cut | `clip_plane` (keep a half-space), `slice_plane` (cross-section curves) |
| Mesh → points | `sample_points_uniformly` |
| UVs and textures | `compute_uvatlas`, `bake_vertex_attr_textures`, `bake_triangle_attr_textures`, `project_images_to_albedo` |
| Surface from a point cloud | legacy only — `geometry.TriangleMesh.create_from_point_cloud_poisson` / `_alpha_shape` / `_ball_pivoting` |

### Registration

| Task | Call |
|---|---|
| Refine a roughly-aligned pair | `multi_scale_icp` (preferred) or `icp` |
| Pick the error metric | `TransformationEstimationPointToPlane` (needs normals, converges fastest), `…PointToPoint`, `…ForColoredICP` (needs colors), `…Symmetric`, `…ForDopplerICP` |
| Downweight outliers | pass a `robust_kernel` to the estimation method |
| Coarse / global alignment | `compute_fpfh_feature` + `correspondences_from_features`; full RANSAC/FGR is **legacy only** (`pipelines.registration.registration_ransac_based_on_feature_matching`, `registration_fgr_based_on_feature_matching`) |
| Score an alignment | `evaluate_registration` |
| Covariance for a pose graph | `get_information_matrix` |
| Optimize many fragments jointly | legacy only — `pipelines.registration.PoseGraph` + `global_optimization` |

### Reconstruction

| Task | Call |
|---|---|
| Fuse RGB-D frames into a volume | `t.geometry.VoxelBlockGrid` — `compute_unique_block_coordinates`, then `integrate` |
| Get the surface out | `extract_point_cloud`, `extract_triangle_mesh` |
| Frame-to-frame camera pose | `t.pipelines.odometry.rgbd_odometry_multi_scale` |
| Online dense SLAM | `t.pipelines.slam.Model` + `Frame` |
| Refine fragment poses | `t.pipelines.slac.run_slac_optimizer_for_fragments` |

### Queries and search

| Task | Call |
|---|---|
| k nearest neighbors | `core.nns.NearestNeighborSearch` — `knn_index()` then `knn_search` |
| All neighbors within a radius | `fixed_radius_index(r)` then `fixed_radius_search` |
| A different radius per query | `multi_radius_index` / `multi_radius_search` |
| k neighbors capped by a radius | `hybrid_index` / `hybrid_search` |
| Ray/mesh intersection | `t.geometry.RaycastingScene.cast_rays` |
| Distance to a surface | `compute_distance` (unsigned), `compute_signed_distance` (signed) |
| Inside/outside test | `compute_occupancy` |
| Closest point on a surface | `compute_closest_points` |
| Visibility / counting hits | `test_occlusions`, `count_intersections`, `list_intersections` |

`KDTreeFlann` is legacy — always use `core.nns` instead.

### Visualization

| Task | Call |
|---|---|
| Just look at it | `visualization.draw(geometry)` |
| Several named, toggleable objects | `visualization.draw([{"name": …, "geometry": …}, …])` or `O3DVisualizer` |
| An image file, no window | `rendering.OffscreenRenderer` → `render_to_image` |
| Custom app with widgets | `gui.Application` + `gui.Window` + `gui.SceneWidget` + `rendering.Open3DScene` |
| Control appearance | `rendering.MaterialRecord` — set `.shader` to `defaultLit`, `defaultUnlit`, `normals`, `depth`, or `unlitLine` |
| Inline in a notebook | `visualization.draw_plotly` |
| Log 3D data over training steps | `visualization.tensorboard_plugin` (needs `tensorboard`) |

## Tensor ↔ Legacy

Same class name, different namespace: `PointCloud`, `TriangleMesh`, `LineSet`,
`Image`, `RGBDImage`, `AxisAlignedBoundingBox`, `OrientedBoundingBox`, and every
`read_*` / `write_*` IO function (`o3d.io.X` → `o3d.t.io.X`).

Renamed or restructured:

| Legacy | Tensor |
|---|---|
| `pipelines.registration.registration_icp` | `t.pipelines.registration.icp` / `multi_scale_icp` |
| `get_information_matrix_from_point_clouds` | `t.pipelines.registration.get_information_matrix` |
| `pipelines.odometry.compute_rgbd_odometry` | `t.pipelines.odometry.rgbd_odometry_multi_scale` |
| `pipelines.integration.ScalableTSDFVolume` | `t.geometry.VoxelBlockGrid` (block-sparse; not drop-in) |
| `geometry.KDTreeFlann` + `KDTreeSearchParam*` | `core.nns.NearestNeighborSearch` — **always migrate** |
| `pcd.points` / `.colors` / `.normals` | `pcd.point["positions" / "colors" / "normals"]` |
| `visualization.draw_geometries` / `Visualizer` | `visualization.draw` / `O3DVisualizer` / `gui` + `rendering` |

### Legacy-only — a fallback here is justified

Octree · TetraMesh · HalfEdgeTriangleMesh · `geometry.keypoint` ·
TSDF volumes (`pipelines.integration.*`) · pose graphs and `global_optimization` ·
RANSAC and FGR global registration · `pipelines.color_map.*` · `camera.*` ·
surface reconstruction from point clouds (Poisson / alpha shape / ball pivoting) ·
visualizer editing and vertex-selection subclasses · `detect_planar_patches` ·
`compute_point_cloud_distance` · `voxel_down_sample_and_trace`

### Tensor-only — no legacy equivalent

`core.Tensor` and explicit device placement · `core.HashMap` / `HashSet` ·
`core.nns.NearestNeighborSearch` · `RaycastingScene` · `VoxelBlockGrid` ·
custom per-point attributes via `TensorMap` · Chamfer/Hausdorff/F-score metrics ·
SLAC and SLAM · Doppler ICP

## Non-Obvious Behavior

Things that cost debugging time and are not visible in a signature.

| API | What to know |
|---|---|
| dtype constants | Lowercase in Python (`o3d.core.float32`); `Float32` is C++ only |
| linear algebra | There is no `o3d.core.linalg` — `matmul`, `inv`, `det`, `svd`, `solve`, `lstsq`, `lu` are `Tensor` methods |
| `from_legacy` | Defaults to `Float32` even though legacy geometry stores `float64` |
| tensor geometry attributes | `pcd.point` is a `TensorMap` with **no `.keys()`** — use `.primary_key` or `dir()` |
| `multi_scale_icp` | `voxel_sizes` and `max_correspondence_distances` must be `o3d.utility.DoubleVector`, not lists |
| any registration result | `.transformation` is always `Float64` on `CPU:0`, whatever device the inputs were on |
| `extract_point_cloud` / `extract_triangle_mesh` | `weight_threshold=3.0` by default — a voxel needs ≥3 observations, so few-frame integrations come back empty |
| `RaycastingScene` | CPU and SYCL only, never CUDA. Device is a constructor argument |
| `compute_signed_distance` | The sign is only meaningful for watertight meshes |
| `fixed_radius_search` | Ragged output: values plus row-split offsets, not a rectangular array |
| `estimate_normals` | Orientation is arbitrary; follow with an `orient_normals_*` call |
| index tensors | Must be `Int64` |
| `Tensor.numpy()` | The tensor must already be on CPU |
| `from_dlpack` | Shares memory; a `to_dlpack` capsule can be consumed only once |
| `o3d.data` | Downloads on first use into `~/open3d_data`; requires network access |
| `gui` | `Application.instance.initialize()` before any window; main thread only |
| optional modules | `webrtc_server`, `ml.torch`, `ml.tf`, `tensorboard_plugin` may be absent — guard with `importlib.util.find_spec` |

## Sample Data

`o3d.data` classes download on construction; read `.path`, `.paths`, or
type-specific attributes (`.depth_paths`, `.color_paths`, `.camera_intrinsic_path`).

| Need | Use |
|---|---|
| A point cloud | `PCDPointCloud`, `PLYPointCloud`, `EaglePointCloud` |
| A mesh | `BunnyMesh`, `ArmadilloMesh`, `KnotMesh` |
| Two clouds to register | `DemoICPPointClouds`, `DemoColoredICPPointClouds`, `DemoFeatureMatchingPointClouds` |
| An RGB-D sequence | `SampleRedwoodRGBDImages`, `SampleFountainRGBDImages`, `BedroomRGBDImages`, `LoungeRGBDImages` |
| A single RGB-D frame | `SampleNYURGBDImage`, `SampleSUNRGBDImage`, `SampleTUMRGBDImage` |
| A textured model | `FlightHelmetModel`, `DamagedHelmetModel`, `MonkeyModel`, `AvocadoModel`, `CrateModel`, `SwordModel` |
| A PBR texture set | `WoodTexture`, `TilesTexture`, `MetalTexture`, `PaintedPlasterTexture`, `TerrazzoTexture`, `WoodFloorTexture` |
| A full indoor scene | `LivingRoomPointClouds`, `OfficePointClouds`, `RedwoodIndoorLivingRoom1/2`, `RedwoodIndoorOffice1/2` |

Full list: `python -c "import open3d as o3d; print(dir(o3d.data))"`

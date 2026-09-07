---
name: open3d-python
description: 'Discover and use the Open3D Python API correctly. Use when writing, reviewing, debugging, or porting Python code that imports open3d — point clouds, triangle meshes, RGB-D images, registration/ICP, odometry, SLAM/SLAC, reconstruction, ray casting, voxel block grids, nearest-neighbor search, mesh/point-cloud IO, GUI and rendering. Covers the tensor API (open3d.core, open3d.t.geometry, open3d.t.io, open3d.t.pipelines; CPU/CUDA/SYCL) versus the legacy Eigen API (open3d.geometry, open3d.io, open3d.pipelines), and the Filament visualization stack (open3d.visualization.gui, open3d.visualization.rendering, draw, O3DVisualizer) versus legacy draw_geometries/Visualizer. Key phrases: open3d python api, o3d.t.geometry, core.Tensor, from_legacy, to_legacy, multi_scale_icp, RaycastingScene, VoxelBlockGrid, O3DVisualizer, open3d stubs, which open3d API should I use.'
---

# Open3D Python API

Find the right Open3D Python API, verify it against the installed package, and
write code that follows current Open3D direction: **tensor API first, Filament
visualization first**.

Applies to Open3D **v0.20.0**.

## When to Use

- Writing or reviewing any Python code that imports `open3d`
- Choosing between `open3d.geometry` (legacy) and `open3d.t.geometry` (tensor)
- Looking up a signature, dtype requirement, default value, or return type
- Porting legacy Eigen-based code to the tensor API
- Debugging dtype / device / attribute errors or GUI threading problems

## API Direction (decide this first)

| Need | Use | Not |
|---|---|---|
| Geometry, IO, registration, odometry, reconstruction | `open3d.core`, `open3d.t.geometry`, `open3d.t.io`, `open3d.t.pipelines` | `open3d.geometry`, `open3d.io`, `open3d.pipelines` |
| Visualization | `open3d.visualization.draw`, `O3DVisualizer`, `visualization.gui`, `visualization.rendering` | `draw_geometries`, `Visualizer`, `VisualizerWith*` |
| CPU + CUDA + SYCL portability | tensor API with an explicit `o3d.core.Device` | legacy (CPU-only, Eigen) |

**Fall back to the legacy API only when there is no tensor equivalent.** When you
do, say so in a comment and isolate the fallback — convert at the boundary with
`from_legacy()` / `to_legacy()`. Legacy-only functionality is listed in
[references/api-map.md](./references/api-map.md).

Never add a silent CPU fallback for a CUDA/SYCL path, and never move large data
between devices without the caller asking for it.

## Discovery Procedure

Work top-down and **stop as soon as you have the answer**. Steps 1–2 reflect the
package that is actually installed and are authoritative.

### 1. Runtime introspection — always start here

```bash
python -c "import open3d as o3d; print(o3d.__version__)"
python -c "import open3d as o3d; print(sorted(n for n in dir(o3d.t.geometry) if not n.startswith('_')))"

# Full docstring + every pybind11 overload
python -c "import open3d as o3d; help(o3d.t.pipelines.registration.multi_scale_icp)"
python -m pydoc open3d.t.geometry.PointCloud
```

`help()` renders the C++ documentation and all overload signatures — this is the
primary API reference. **Do not use `inspect.signature()`** on pybind11 methods;
it raises `ValueError: no signature found for builtin ...`.

Confirm availability before use, since optional features vary by wheel:

```python
o3d.core.cuda.is_available()      # also o3d.core.sycl.is_available()
```

### 2. Typing stubs (`.pyi`) — best for overloads and grepping

The wheel ships `py.typed` and `.pyi` stubs carrying full signatures *and*
docstrings, so static greps are reliable:

```bash
STUBS=$(python -c "import open3d,os;print(os.path.dirname(open3d.__file__))")/pybind
grep -rn "def multi_scale_icp\|class PointCloud\|def from_legacy" "$STUBS" --include='*.pyi'
```

Stub layout: `core/{__init__,cuda,sycl,nns,kernel}.pyi`, `t/{geometry,io}.pyi`,
`t/pipelines/{registration,odometry,slac,slam}.pyi`,
`visualization/{__init__,gui,rendering}.pyi`, `geometry.pyi`, `io.pyi`, `pipelines/`.

### 3. Examples bundled with the wheel — check these before writing new code

The wheel ships runnable examples. **Prefer these over any other example source.**

```bash
open3d example --list                                   # all categories
open3d example --list geometry                          # one category
open3d example --show geometry/point_cloud_convex_hull  # print source
open3d example geometry/point_cloud_convex_hull         # run it
```

Categories: `camera`, `geometry`, `io`, `pipelines`, `utility`, `visualization`.
To read them directly:

```bash
python -c "import open3d,os;print(os.path.join(os.path.dirname(open3d.__file__),'examples'))"
```

That directory also contains `reconstruction_system/` and
`t_reconstruction_system/` (multi-file pipelines, not runnable via the CLI).

### 4. Tutorials and remaining examples — online

- Tutorials and API reference: <https://www.open3d.org/docs/latest/> — use the
  site's search box (docs are versioned; `latest` tracks development)
- Source, issues, full example tree: <https://github.com/isl-org/Open3D> — use
  GitHub code search, e.g. `repo:isl-org/Open3D multi_scale_icp path:examples/`

Use these for conceptual tutorials and for examples not shipped in the wheel.

## Verified Snippets

All snippets below were executed against a real Open3D build and produced the
stated results. Note `o3d.core.float32` is **lowercase** in Python (C++ uses
`core::Float32`).

### Read, inspect, write

```python
import open3d as o3d

pcd = o3d.t.io.read_point_cloud(o3d.data.DemoICPPointClouds().paths[0])
print(pcd.point.primary_key)              # 'positions'
print(pcd.point["positions"].shape)       # SizeVector[198835, 3]
o3d.t.io.write_point_cloud("out.ply", pcd)
```

### Build geometry from NumPy, on a chosen device

```python
import numpy as np
import open3d as o3d

device = (o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available()
          else o3d.core.Device("CPU:0"))
positions = o3d.core.Tensor(np.random.rand(100, 3).astype(np.float32), device=device)
pcd = o3d.t.geometry.PointCloud(positions)
pcd.point["colors"] = o3d.core.Tensor.ones((100, 3), o3d.core.float32, device)
```

### Downsample, estimate normals, remove outliers

```python
down = pcd.voxel_down_sample(0.05)
down.estimate_normals(max_nn=30, radius=0.1)
clean, mask = down.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
```

### Point-to-plane multi-scale ICP

The vector arguments must be `o3d.utility.DoubleVector`, not plain lists.

```python
import open3d as o3d
reg = o3d.t.pipelines.registration

data = o3d.data.DemoICPPointClouds()
source = o3d.t.io.read_point_cloud(data.paths[0])
target = o3d.t.io.read_point_cloud(data.paths[1])
source.estimate_normals(); target.estimate_normals()

result = reg.multi_scale_icp(
    source, target,
    o3d.utility.DoubleVector([0.05, 0.025, 0.0125]),   # voxel sizes, coarse -> fine
    [reg.ICPConvergenceCriteria(max_iteration=n) for n in (30, 15, 10)],
    o3d.utility.DoubleVector([0.1, 0.05, 0.025]),      # max correspondence distances
    estimation_method=reg.TransformationEstimationPointToPlane())
print(result.fitness, result.inlier_rmse)   # 0.1989 0.011
# result.transformation is always Float64 on CPU:0
```

### Feature-based correspondences (FPFH)

```python
reg = o3d.t.pipelines.registration
src_fpfh = reg.compute_fpfh_feature(source, max_nn=100, radius=0.25)
dst_fpfh = reg.compute_fpfh_feature(target, max_nn=100, radius=0.25)
corres = reg.correspondences_from_features(src_fpfh, dst_fpfh)   # (N, 2) Int64
```

RANSAC global registration is legacy-only — use
`o3d.pipelines.registration.registration_ransac_based_on_feature_matching`.

### Ray casting, distance, and signed distance

```python
import open3d as o3d

mesh = o3d.t.geometry.TriangleMesh.create_sphere(1.0)
scene = o3d.t.geometry.RaycastingScene()          # CPU; pass device= for SYCL
scene.add_triangles(mesh)

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=60, center=[0, 0, 0], eye=[0, 0, 3], up=[0, 1, 0],
    width_px=32, height_px=32)
ans = scene.cast_rays(rays)                       # dict: t_hit, geometry_ids, ...

query = o3d.core.Tensor([[0., 0, 0], [2, 0, 0]], o3d.core.float32)
print(scene.compute_signed_distance(query).numpy())   # [-0.9939  1.0]
```

On an Intel GPU with a SYCL wheel, build the scene on the device:

```python
scene = o3d.t.geometry.RaycastingScene(device=o3d.core.Device("SYCL:0"))
```

### TSDF integration with VoxelBlockGrid

```python
import numpy as np
import open3d as o3d

data = o3d.data.SampleRedwoodRGBDImages()
intrinsic = o3d.core.Tensor(
    o3d.io.read_pinhole_camera_intrinsic(data.camera_intrinsic_path).intrinsic_matrix,
    o3d.core.float64)
extrinsic = o3d.core.Tensor(np.eye(4), o3d.core.float64)   # use real per-frame poses

vbg = o3d.t.geometry.VoxelBlockGrid(
    attr_names=("tsdf", "weight", "color"),
    attr_dtypes=(o3d.core.float32,) * 3,
    attr_channels=((1,), (1,), (3,)),
    voxel_size=3.0 / 512, block_resolution=16, block_count=50000)

for depth_path, color_path in zip(data.depth_paths, data.color_paths):
    depth = o3d.t.io.read_image(depth_path)
    color = o3d.t.io.read_image(color_path)
    blocks = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1000.0, 3.0)
    vbg.integrate(blocks, depth, color, intrinsic, intrinsic, extrinsic, 1000.0, 3.0)

# weight_threshold defaults to 3.0: a voxel needs >=3 observations to survive.
# Lower it when integrating only one or two frames, or you get an empty result.
pcd = vbg.extract_point_cloud(weight_threshold=1.0)
mesh = vbg.extract_triangle_mesh(weight_threshold=1.0)
```

### Nearest neighbor search

```python
import numpy as np
import open3d as o3d

points = o3d.core.Tensor(np.random.rand(1000, 3).astype(np.float32))
queries = points[:5]
nns = o3d.core.nns.NearestNeighborSearch(points)

nns.knn_index()
indices, squared_dists = nns.knn_search(queries, 8)          # (5, 8)

nns.fixed_radius_index(0.2)                                  # radius given at index time
idx, dist2, splits = nns.fixed_radius_search(queries, 0.2)   # ragged + row-split offsets
```

### Mesh operations

```python
mesh = o3d.t.io.read_triangle_mesh(o3d.data.KnotMesh().path)
mesh.compute_vertex_normals()
simplified = mesh.simplify_quadric_decimation(target_reduction=0.5)
samples = mesh.sample_points_uniformly(1000)
union = mesh.boolean_union(o3d.t.geometry.TriangleMesh.create_sphere(0.1))
```

### Tensor ↔ legacy and device transfer

```python
tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd, o3d.core.float32)
legacy_pcd = tensor_pcd.to_legacy()

gpu = tensor_pcd.to(o3d.core.Device("CUDA:0"))   # explicit; nothing moves implicitly
cpu = gpu.cpu()
array = tensor_pcd.point["positions"].numpy()    # must already be on CPU
```

### Visualization

```python
import open3d as o3d

o3d.visualization.draw(mesh)                     # one-liner; tensor or legacy geometry
```

Offscreen rendering, for deterministic images and headless use:

```python
import open3d.visualization.rendering as rendering

renderer = rendering.OffscreenRenderer(640, 480)
material = rendering.MaterialRecord()
material.shader = "defaultLit"                   # or defaultUnlit, normals, depth
renderer.scene.add_geometry("mesh", mesh, material)
renderer.setup_camera(60.0, mesh.get_axis_aligned_bounding_box(), [0, 0, 0])
o3d.io.write_image("render.png", renderer.render_to_image())
```

Full GUI application:

```python
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

app = gui.Application.instance
app.initialize()                                 # required before creating windows
window = app.create_window("Open3D", 1024, 768)

widget = gui.SceneWidget()
widget.scene = rendering.Open3DScene(window.renderer)
widget.scene.add_geometry("mesh", mesh, rendering.MaterialRecord())
window.add_child(widget)
widget.setup_camera(60.0, widget.scene.bounding_box, [0, 0, 0])

app.run()                                        # main thread only
```

See `open3d example --list visualization` for `draw`, `vis_gui`, `demo_scene`,
`render_to_image`, `add_geometry`, and `multiple_windows`.

## Gotchas

| Symptom | Cause / fix |
|---|---|
| `AttributeError: ... has no attribute 'Float32'` | Python dtypes are lowercase: `o3d.core.float32`. Only C++ uses `core::Float32`. Similarly `o3d.core.linalg` does not exist — linear algebra lives on `Tensor` (`matmul`, `inv`, `det`, `svd`, `solve`, `lstsq`, `lu`). |
| `KeyError: Key ... not found in TensorMap` | Attribute not set. `TensorMap` has no `.keys()`; use `pcd.point.primary_key` or `dir(pcd.point)`. |
| `TypeError: incompatible function arguments` on `multi_scale_icp` | Pass `o3d.utility.DoubleVector([...])` for `voxel_sizes` and `max_correspondence_distances`, not plain lists. |
| `extract_point_cloud()` returns 0 points | `weight_threshold` defaults to `3.0`, so a voxel needs ≥3 overlapping observations. Integrate more frames or pass `weight_threshold=1.0`. |
| dtype error in a tensor op | Tensor geometry defaults to `Float32`; legacy is `float64`. Indices are `Int64`. Registration transforms are always CPU `Float64`. Pass dtype explicitly. |
| device mismatch error | Every input to a tensor pipeline must be on the same device. `.to(device)` each one — converting one object does not move the others. |
| `TypeError` passing a point cloud to ICP | `o3d.geometry.PointCloud` and `o3d.t.geometry.PointCloud` are distinct types, as are `o3d.io.read_point_cloud` and `o3d.t.io.read_point_cloud`. Convert explicitly. |
| `ValueError: no signature found` | `inspect.signature()` on a pybind11 method. Use `help()` or the `.pyi` stub. |
| `RaycastingScene` fails on GPU | It supports CPU and SYCL only, not CUDA. Pass `device=` at construction; move inputs to a supported device. |
| `ModuleNotFoundError` for a documented module | `webrtc_server`, `ml.torch`, `ml.tf`, `tensorboard_plugin` are build/dependency dependent. The docs cover more than any single wheel ships — guard with `importlib.util.find_spec`. |
| GUI hangs, crashes, or renders nothing | `gui.Application.instance.initialize()` not called, or GUI touched off the main thread. For headless, use `rendering.OffscreenRenderer`. |
| Dataset download fails | `o3d.data` fetches on first use and caches under `~/open3d_data`. Requires network access. |
| Unexpected aliasing after `from_dlpack` | `Tensor.from_dlpack()` shares memory and a `to_dlpack()` capsule is single-use. Copy semantics of `Tensor(...)` vs `Tensor.from_numpy(...)` differ — verify before relying on mutation. |

## Installing

```bash
pip install open3d        # CUDA-enabled on x86_64 Linux; CPU on Windows/macOS
pip install open3d-cpu    # smaller CPU-only wheel (x86_64 Linux)
pip install open3d-xpu    # Intel GPU / SYCL wheel (x86_64 Linux and Windows)
pip install open3d-cuda   # CUDA wheel on Windows
pip install open3d[ml]    # add the ml extra for open3d.ml.torch / open3d.ml.tf
```

All variants expose the same `open3d` module name. Prefer a prebuilt wheel —
building from source takes a long time. Details and development wheels:
<https://www.open3d.org/docs/latest/getting_started.html>

## This Skill

This skill ships inside the wheel at
`<site-packages>/open3d/agent_skills/open3d-python`. To make it available to an
agent in another project:

```bash
open3d agent_skill --path                    # where it lives
open3d agent_skill --install .github/skills  # copy into a project
open3d agent_skill                           # print it to stdout
```

## Reference Files

- [references/api-map.md](./references/api-map.md) — module map, per-class method
  inventories, tensor↔legacy mapping, legacy-only and tensor-only lists

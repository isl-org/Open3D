---
name: open3d-cpp
description: 'Discover and use the Open3D C++ API correctly. Use when writing, reviewing, debugging, or porting C++ code against Open3D — point clouds, triangle meshes, RGB-D images, registration/ICP, odometry, SLAM/SLAC, reconstruction, ray casting, voxel block grids, nearest-neighbor search, geometry IO, GUI and rendering — or when linking Open3D from a CMake project. Covers the tensor API (open3d::core, open3d::t::geometry, open3d::t::io, open3d::t::pipelines; CPU/CUDA/SYCL) versus the legacy Eigen API (open3d::geometry, open3d::io, open3d::pipelines), and the Filament stack (open3d::visualization::gui, ::rendering, O3DVisualizer) versus the legacy OpenGL Visualizer. Key phrases: open3d c++ api, core::Tensor, t::geometry::PointCloud, FromLegacy, ToLegacy, Open3DScene, MaterialRecord, find_package Open3D, Open3D::Open3D, open3d-devel, which open3d header.'
---

# Open3D C++ API

Find the right Open3D C++ symbol, verify it against the installed headers, and
write code that follows current Open3D direction: **tensor API first, Filament
visualization first**.

Applies to Open3D **v0.20.0**. This skill covers the **public API for
applications that use Open3D**, not Open3D's own internals.

## When to Use

- Writing or reviewing C++ that includes `open3d/...` headers
- Choosing between `open3d::geometry` (legacy) and `open3d::t::geometry` (tensor)
- Looking up a signature, dtype/device contract, or class hierarchy
- Linking Open3D from a CMake project

## API Direction (decide this first)

| Need | Use | Not |
|---|---|---|
| Geometry, IO, registration, odometry, reconstruction | `open3d::core`, `open3d::t::geometry`, `open3d::t::io`, `open3d::t::pipelines` | `open3d::geometry`, `open3d::io`, `open3d::pipelines` |
| Visualization | `open3d::visualization::gui`, `::rendering`, `Draw`, `O3DVisualizer` | `visualization::Visualizer`, `DrawGeometries` |
| CPU + CUDA + SYCL portability | `core::Tensor` with an explicit `core::Device` | legacy Eigen (CPU-only) |

**Fall back to the legacy API only when there is no tensor equivalent** (see the
legacy-only list in [references/api-map.md](./references/api-map.md)). Convert at
the boundary with `FromLegacy()` / `ToLegacy()` and note the reason in a comment.

Never add a silent CPU fallback for a CUDA/SYCL path, and never move data between
devices implicitly.

## Get Open3D

**Do not build Open3D from source unless you have to — it takes a long time.**
Download a prebuilt binary package instead:

- Releases: <https://github.com/isl-org/Open3D/releases> — `open3d-devel-*.tar.xz`
  for Linux, macOS, and Windows, in CPU and CUDA variants
- Extract it and point CMake at it with `-DOpen3D_ROOT=/path/to/open3d-devel-...`

If you genuinely must build from source, follow
<https://www.open3d.org/docs/latest/compilation.html> rather than improvising.

This skill ships inside that package at `share/Open3D/agent_skills/open3d-cpp`
(`bin/Open3D/agent_skills/open3d-cpp` on Windows). Copy it into your project's
`.github/skills/` to make it available to an agent.

## Discovery Procedure

The installed headers are authoritative — they match the binary you are linking
against. Doxygen HTML is generally **not** installed locally; use the website.

### 1. Grep the installed headers — start here

```bash
O3D=/path/to/open3d-devel-.../include        # or /usr/local/include
grep -rn "class PointCloud" "$O3D/open3d/t/geometry/"
grep -rn "FromLegacy\|ToLegacy" "$O3D/open3d/t/geometry/PointCloud.h"
grep -rn "MultiScaleICP" "$O3D/open3d/t/pipelines/registration/"
```

Declarations carry Doxygen `\brief`, `\param`, and `\return` comments — that is
the primary C++ reference. `rg` works equally well. The umbrella header
`open3d/Open3D.h` lists every public sub-header and is the quickest way to see
the whole surface.

To find a class when you do not know its header:

```bash
grep -rln "class RaycastingScene" "$O3D/open3d/"
```

### 2. Online API reference and tutorials

- C++ API reference: <https://www.open3d.org/docs/latest/cpp_api/> — use the
  site's search box
- Tutorials and conceptual docs: <https://www.open3d.org/docs/latest/>
- Source, issues, and the full example tree:
  <https://github.com/isl-org/Open3D> — use GitHub code search, e.g.
  `repo:isl-org/Open3D MultiScaleICP path:examples/cpp`

### 3. Examples on GitHub

C++ examples live in
[`examples/cpp/`](https://github.com/isl-org/Open3D/tree/main/examples/cpp).
Useful entry points: `PointCloud.cpp`, `TriangleMesh.cpp`, `TICP.cpp`,
`RegistrationRANSAC.cpp`, `TIntegrateRGBD.cpp`, `Draw.cpp`,
`OffscreenRendering.cpp`, `MultipleWindows.cpp`, `Visualizer.cpp`.

If you also have Open3D's Python wheel installed, `open3d example --list` gives
runnable Python equivalents that mirror the C++ API one-to-one (`snake_case`
instead of `PascalCase`).

## Linking Open3D

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyApp LANGUAGES CXX)

find_package(Open3D REQUIRED)

add_executable(MyApp main.cpp)
target_link_libraries(MyApp PRIVATE Open3D::Open3D)
```

```bash
cmake -S . -B build -DOpen3D_ROOT=/path/to/open3d-devel-...
cmake --build build
```

Open3D requires a **C++17-capable** compiler and CMake 3.24+. Working templates:
[`examples/cmake/open3d-cmake-find-package`](https://github.com/isl-org/Open3D/tree/main/examples/cmake/open3d-cmake-find-package)
(prebuilt package — recommended) and
[`examples/cmake/open3d-cmake-external-project`](https://github.com/isl-org/Open3D/tree/main/examples/cmake/open3d-cmake-external-project)
(builds Open3D alongside your project — slow).

`pkg-config` also works on Linux/macOS with shared libraries, but CMake is
strongly preferred because it handles the optional backends correctly:

```bash
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:<install>/lib/pkgconfig"
c++ main.cpp -o app $(pkg-config --cflags --libs Open3D)   # libs must follow sources
```

Full details: <https://www.open3d.org/docs/latest/cpp_project.html>

## Snippets

Adapted from the official examples; see each link for the full program.

### Read, process, write a point cloud

```cpp
#include "open3d/Open3D.h"
using namespace open3d;

t::geometry::PointCloud pcd;
t::io::ReadPointCloud("input.ply", pcd);

auto down = pcd.VoxelDownSample(0.05);
down.EstimateNormals(30, 0.1);
auto [clean, mask] = down.RemoveStatisticalOutliers(20, 2.0);

t::io::WritePointCloud("output.ply", clean);
utility::LogInfo("{} -> {} points", pcd.GetPointPositions().GetLength(),
                 clean.GetPointPositions().GetLength());
```

### Tensors and devices

```cpp
core::Device device = core::cuda::IsAvailable() ? core::Device("CUDA:0")
                                                : core::Device("CPU:0");
core::Tensor points = core::Tensor::Zeros({100, 3}, core::Float32, device);

// From existing memory (copies into an Open3D-owned blob)
std::vector<float> raw{0, 0, 0, 1, 0, 0, 0, 1, 0};
core::Tensor from_raw(raw, {3, 3}, core::Float32, device);

t::geometry::PointCloud pcd(points);
pcd.SetPointAttr("colors", core::Tensor::Ones({100, 3}, core::Float32, device));

auto on_cpu = pcd.To(core::Device("CPU:0"));   // explicit; nothing moves implicitly
```

C++ dtype constants are capitalized (`core::Float32`, `core::Int64`); the Python
equivalents are lowercase.

### Point-to-plane multi-scale ICP

```cpp
using namespace open3d::t::pipelines::registration;

std::vector<double> voxel_sizes{0.05, 0.025, 0.0125};
std::vector<double> max_correspondence_distances{0.1, 0.05, 0.025};
std::vector<ICPConvergenceCriteria> criteria;
criteria.emplace_back(1e-6, 1e-6, 30);
criteria.emplace_back(1e-6, 1e-6, 15);
criteria.emplace_back(1e-6, 1e-6, 10);

source.EstimateNormals();
target.EstimateNormals();

auto result = MultiScaleICP(
        source, target, voxel_sizes, criteria, max_correspondence_distances,
        core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        TransformationEstimationPointToPlane());
utility::LogInfo("fitness {} rmse {}", result.fitness_, result.inlier_rmse_);
```

`result.transformation_` is always `Float64` on `CPU:0`. Source:
[TICP.cpp](https://github.com/isl-org/Open3D/blob/main/examples/cpp/TICP.cpp).

### Ray casting and signed distance

```cpp
auto mesh = t::geometry::TriangleMesh::CreateSphere(1.0);
t::geometry::RaycastingScene scene;            // or scene(0, core::Device("SYCL:0"))
scene.AddTriangles(mesh);

auto rays = t::geometry::RaycastingScene::CreateRaysPinhole(
        60.0, core::Tensor::Init<float>({0, 0, 0}),   // center
        core::Tensor::Init<float>({0, 0, 3}),         // eye
        core::Tensor::Init<float>({0, 1, 0}),         // up
        640, 480);
auto result = scene.CastRays(rays);            // map: "t_hit", "geometry_ids", ...

auto query = core::Tensor::Init<float>({{0, 0, 0}, {2, 0, 0}});
auto sdf = scene.ComputeSignedDistance(query);
```

### TSDF integration

```cpp
t::geometry::VoxelBlockGrid vbg(
        {"tsdf", "weight", "color"},
        {core::Float32, core::Float32, core::Float32},
        {{1}, {1}, {3}}, 3.0f / 512, 16, 50000, device);

for (size_t i = 0; i < depth_files.size(); ++i) {
    auto depth = t::io::CreateImageFromFile(depth_files[i])->To(device);
    auto color = t::io::CreateImageFromFile(color_files[i])->To(device);
    auto blocks = vbg.GetUniqueBlockCoordinates(*depth, intrinsic, extrinsics[i],
                                                1000.0f, 3.0f);
    vbg.Integrate(blocks, *depth, *color, intrinsic, intrinsic, extrinsics[i],
                  1000.0f, 3.0f);
}
auto pcd = vbg.ExtractPointCloud();            // weight threshold defaults to 3.0
auto mesh = vbg.ExtractTriangleMesh();
```

Source: [TIntegrateRGBD.cpp](https://github.com/isl-org/Open3D/blob/main/examples/cpp/TIntegrateRGBD.cpp).

### Nearest neighbor search

```cpp
core::Tensor points = core::Tensor::Init<float>({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
core::nns::NearestNeighborSearch nns(points);

nns.KnnIndex();
auto [indices, distances2] = nns.KnnSearch(points, 2);

nns.FixedRadiusIndex(0.5);
auto [r_idx, r_dist2, r_splits] = nns.FixedRadiusSearch(points, 0.5);
```

### Visualization

```cpp
auto mesh = std::make_shared<geometry::TriangleMesh>();
io::ReadTriangleMesh("mesh.ply", *mesh);
mesh->ComputeVertexNormals();
visualization::Draw({mesh});                    // one-liner, Filament-backed
```

Named objects and per-object visibility:

```cpp
visualization::Draw({visualization::DrawObject("source", source),
                     visualization::DrawObject("target", target, false)});
```

Offscreen rendering:

```cpp
using namespace open3d::visualization;

auto &app = gui::Application::GetInstance();
app.Initialize();

auto *renderer = new rendering::FilamentRenderer(
        rendering::EngineInstance::GetInstance(), 640, 480,
        rendering::EngineInstance::GetResourceManager());
auto *scene = new rendering::Open3DScene(*renderer);

rendering::MaterialRecord material;
material.shader = "defaultLit";
scene->AddGeometry("mesh", mesh.get(), material);

auto image = app.RenderToImage(*renderer, scene->GetView(), scene->GetScene(),
                               640, 480);
io::WriteImage("render.png", *image);
```

Sources: [Draw.cpp](https://github.com/isl-org/Open3D/blob/main/examples/cpp/Draw.cpp),
[OffscreenRendering.cpp](https://github.com/isl-org/Open3D/blob/main/examples/cpp/OffscreenRendering.cpp).

## Conventions and Gotchas

- **Naming**: C++ is `PascalCase` where Python is `snake_case`
  (`VoxelDownSample` ↔ `voxel_down_sample`). Getters are explicit in C++:
  `GetPointPositions()` / `SetPointAttr()` versus Python's `pcd.point["positions"]`.
- **Tensors** are shape + strides + blob + dtype + device. Copies are shallow by
  default; use `Clone()` for a deep copy and `Contiguous()` when a kernel needs it.
  Dtype constants are `core::Float32`, `core::Float64`, `core::Int64`, `core::Bool`.
- **Device is explicit.** All inputs to a tensor pipeline must be on the same
  device; `To(device)` each one.
- **Indexing** uses `core::TensorKey::Index/Slice/IndexTensor` with `core::None`
  for an open slice bound, following NumPy semantics.
- **`utility::LogError` is `[[noreturn]]`** and throws `std::runtime_error` — it
  is the error mechanism, not just a print. Also `LogWarning`, `LogInfo`,
  `LogDebug`; set the level with `utility::SetVerbosityLevel`.
- **`RaycastingScene` supports CPU and SYCL devices**, not CUDA. Select with the
  constructor's `device` argument. (The class comment in `RaycastingScene.h` still
  says CPU-only; the binding docs and implementation are correct.)
- **`VoxelBlockGrid::ExtractPointCloud/ExtractTriangleMesh`** take a weight
  threshold defaulting to `3.0`; a voxel needs that many observations to appear.
- **Registration transformations** are always `Float64` on `CPU:0`.
- **Legacy and tensor types are distinct** — `geometry::PointCloud` and
  `t::geometry::PointCloud` do not interconvert implicitly.
- **CUDA/SYCL availability** is a property of the binary you downloaded. Check
  with `core::cuda::IsAvailable()` / `core::sycl::IsAvailable()` at runtime.

## Reference Files

- [references/api-map.md](./references/api-map.md) — namespace map with headers,
  per-class method inventories, tensor↔legacy mapping, legacy-only and
  tensor-only lists

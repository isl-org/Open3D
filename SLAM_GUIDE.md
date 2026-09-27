# RGB-D localization and SLAM guide

## 1. What the demo teaches

The demo estimates camera motion and builds a colored point-cloud map from a
synchronized RGB-D sequence. Its default path is autonomous: all estimation and
loop closure use RGB and depth only. A TUM `groundtruth.txt` file is optional in
this mode and is used exclusively for the blue reference path and accuracy KPIs.

The display separates several answers to "where is the camera?":

| Color/object | Meaning |
|---|---|
| Orange path | Local dense RGB-D odometry, never globally corrected |
| Green path | Current/final pose-graph trajectory |
| Red path | Local odometry plus artificial incremental SE(3) noise |
| Blue path | Optional normalized GT reference, not an autonomous estimate |
| Cyan cloud | Current depth scan transformed by the corrected pose |
| RGB cloud | Accumulated keyframe map, rebuilt after graph optimization |
| Magenta sphere / cyan frustum | Corrected camera position and orientation |
| Yellow rings | Heuristic positional uncertainty display |

The cyan scan is a useful qualitative localization check: aligned surfaces should
overlap the RGB map. It is not a quantitative accuracy test because the map was
built from the same estimates.

This remains an educational implementation. The algorithms are real, but the
loop database is linear, mapping is point-cloud concatenation, covariance is a
configured random walk, and the system has no persistent landmarks,
relocalization, IMU fusion, or production loop manager.

## 2. Run all three modes

```bash
cd /home/jacob/airobotics/3d_slam_example
source .venv/bin/activate

# Default: autonomous RGB-D-only SLAM.
python slam_demo.py

# Equivalent explicit selection.
python slam_demo.py --mode autonomous

# Dense sequential odometry and mapping, no loop optimization.
python slam_demo.py --mode odometry-only

# Scheduled GT-derived teaching constraints; groundtruth.txt is required.
python slam_demo.py --mode gt-assisted

# Select gt-assisted and change its interval.
python slam_demo.py --loop-closure-every 20

# Alias for odometry-only.
python slam_demo.py --no-loop-closure

# Headless and unpaced.
python slam_demo.py --no-viewer --no-realtime --max-frames 100
```

| Mode | What creates global constraints | GT role |
|---|---|---|
| `autonomous` | ORB appearance retrieval, depth-backed PnP, ICP verification | Optional evaluation/display only |
| `gt-assisted` | Periodic node-zero-to-current transform computed from GT | Required estimation input; resulting accuracy is not independent |
| `odometry-only` | Nothing; graph contains sequential edges only | Optional evaluation/display only |

Autonomous detection runs only at map keyframes. The scheduled
`loop_closure_every_n_frames` setting applies only to `gt-assisted` mode.

## 3. Dataset and association

Only RGB and depth are required:

```text
dataset/
|-- rgb.txt
|-- depth.txt
|-- rgb/*.png
`-- depth/*.png
```

Optionally add:

```text
groundtruth.txt   # timestamp tx ty tz qx qy qz qw
```

For each RGB timestamp, the loader selects the nearest depth timestamp and keeps
the record when that gap is at most `association_max_dt_s`. If GT exists, the
nearest pose is attached only when its own gap passes the same tolerance. Missing
GT association does not discard valid RGB-D in autonomous or odometry-only mode.
`gt-assisted` later filters out records without GT before applying
`frame_stride` and `max_frames`.

Depth is converted to meters with `depth_raw / depth_scale` and truncated at
`depth_trunc_m`. Optional GT poses are normalized by the first associated pose:

```text
T_map_camera_gt[i] = inverse(T_world_camera_gt[0]) * T_world_camera_gt[i]
```

This normalization supports comparison; it does not define autonomous motion.
The estimated graph independently starts with identity at the first camera.

## 4. Pipeline, step by step

### 4.1 Dense frame-to-frame odometry

`processing_loop()` calls Open3D `compute_rgbd_odometry()` with the hybrid
photometric/geometric Jacobian. It estimates a relative transform from the
previous camera to the current camera by aligning image intensity and depth. The
camera-to-map pose is composed using the inverse:

```text
T_map_current = T_map_previous * inverse(T_current_previous)
```

This local solver is dense. It does not use the ORB descriptors described below;
ORB is only for finding long-range loop candidates. Every processed frame gets a
pose-graph node and a sequential edge. Failed odometry holds the pose and inserts
an identity edge with weak information so failure remains visible.

### 4.2 Keyframes and ORB ratio-match retrieval

Every `map_every_n_frames` processed frames, the program creates both a map
keyframe and an autonomous loop keyframe. ORB detects up to `orb_features`
keypoints. A descriptor is retained only if its rounded image pixel has valid
metric depth below `max_feature_depth_m`; the pixel is then back-projected:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
object_point = [x, y, z] in the old keyframe camera
```

Eligible old keyframes must be at least `min_frame_separation` processed indices
away. After a loop acceptance, no database search occurs for `cooldown_frames`.
For each eligible old/current pair, brute-force Hamming KNN matching obtains two
neighbors per old descriptor. Lowe's ratio filter keeps a match when:

```text
best_distance < ratio_test * second_best_distance
```

The normalized appearance score is:

```text
good_match_count / max(1, min(old_descriptor_count, current_descriptor_count))
```

A candidate must pass `min_matches` and `min_match_score`. Candidates are ranked
by absolute good-match count, not normalized score, and only the strongest
`max_candidates` proceed to geometry. These choices are simple and transparent,
not a scalable bag-of-words or learned retrieval system.

### 4.3 Depth-backed `solvePnPRansac`

For every ratio match, the old keyframe supplies a depth-backed 3D object point
and the current keyframe supplies a 2D image observation. OpenCV
`solvePnPRansac()` with EPNP estimates the transform from the old candidate camera
to the current camera. It uses configured RANSAC iterations, pixel reprojection
threshold, and confidence.

The proposal is rejected unless it reaches both the absolute
`min_pnp_inliers` and relative `min_pnp_inlier_ratio`. PnP provides metric scale
because its object points came from RGB-D depth.

### 4.4 ICP refinement and verification

Open3D point-to-point ICP aligns the old and current local keyframe clouds using
the PnP transform as initialization. Acceptance requires adequate fitness, low
inlier RMSE, and limited translation and rotation change from the PnP proposal:

```text
fitness >= min_icp_fitness
inlier_rmse <= max_icp_rmse_m
||translation(T_icp * inverse(T_pnp))|| <= max_icp_translation_correction_m
angle(T_icp * inverse(T_pnp)) <= max_icp_rotation_correction_deg
```

The final ICP transform and an information matrix computed from the two clouds
form an uncertain source-keyframe-to-current edge. The first candidate to pass is
accepted. Match count, PnP inliers, ICP fitness, and ICP RMSE for accepted loops
are preserved in `telemetry.csv`.

Thresholds reduce, but cannot eliminate, false closures. Repeated geometry,
texture aliasing, sparse depth at keypoints, and local ICP minima remain risks.

### 4.5 Pose-graph optimization

The graph combines certain sequential odometry edges with uncertain loop edges.
After an autonomous verified edge or a scheduled GT-assisted edge is inserted,
Open3D's Levenberg-Marquardt global optimizer adjusts graph nodes while holding
node zero fixed. Its correspondence scale, uncertain-edge pruning threshold, and
loop preference come from `optimization.*`.

The entire green trajectory is refreshed after optimization. Cached keyframe
clouds remain in their camera coordinates, so every cloud can be re-transformed
by its updated graph-node pose and the map can be rebuilt consistently.

The reported correction jump is the current endpoint's translation displacement
before versus after optimization. It does not summarize all node movement or
prove that the accepted edge improved accuracy. The code also does not inspect
whether Open3D pruned the uncertain edge afterward.

### 4.6 Why this is not bundle adjustment

| Method | Variables | Residuals |
|---|---|---|
| Pose graph here | Camera poses | Relative SE(3) constraints |
| Classical bundle adjustment | Camera poses and persistent landmarks | 2D reprojection errors |
| Dense RGB-D odometry | One relative pose | Photometric and depth alignment errors |

ORB matches here are transient loop evidence. They are not maintained as
landmark tracks, so there is no camera-plus-landmark bundle-adjustment problem.

### 4.7 Map and covariance

Each frame cloud is voxel-downsampled and capped. The current scan is displayed
every frame; map keyframes are cached and accumulated. After optimization the map
is rebuilt, and at shutdown it can receive a final global voxel pass before being
written to PLY.

The covariance is deliberately pedagogical:

```text
P[k] = P[k-1] + Q
P_after_closure = loop_closure_shrink * P_before_closure
```

The yellow rings visualize eigenvalue-derived radii from the positional 3x3
block. This is not rigorous SE(3) propagation or a measurement covariance.

## 5. Outputs and analysis

| Output | Use |
|---|---|
| `telemetry.csv` | Per-frame timings, odometry status, search selectivity, accepted-loop verification details, correction jump, and heuristic uncertainty |
| `output/autonomous_map.ply` | Final displayed colored point map after optional output voxel downsampling |
| `output/trajectories.npz` | XYZ arrays for noisy, local, optimized, and optional GT trajectories |
| `output/pose_graph.json` | Open3D graph nodes and sequential/loop edges |
| `output/run_summary.json` | Aggregate tracking, performance, loop, trajectory, map, and optional GT-accuracy KPIs |

`telemetry.csv` is opened in write mode beside `slam_demo.py`; configured writers
replace same-named outputs in `output/` when they run. The PLY is skipped if the
accumulated map is empty, and an interrupted worker may not reach pose-graph
serialization. The NPZ arrays are positions, not full poses. The GT array can be
empty and has no frame-index sidecar.

The README's `## KPIs` section is the authoritative metric reference. Key
interpretation rules are:

- `processing_hz` and JSON compute throughput invert worker compute time; they
  are not end-to-end sustained rates.
- Effective throughput uses run wall time and therefore includes pacing, GUI,
  pauses, and final map/trajectory serialization.
- Loop comparison/candidate/check counts describe detector selectivity, not
  loop precision or recall.
- Autonomous acceptance rate and mean match/PnP/ICP quality summarize only loops
  that passed every threshold; they do not characterize rejected candidates or
  prove that accepted loops are correct.
- ATE compares translations directly in the first-camera frame without fitting
  an additional trajectory alignment.
- Accuracy is omitted without GT. In autonomous mode GT never participates in
  estimation; in GT-assisted mode optimized accuracy is inherently circular.
- The CLI prints a high-value subset; `run_summary.json` has all aggregate fields,
  while `telemetry.csv` retains individual frames and accepted-loop details.

## 6. Parameters worth changing

| Stage | Settings | Tradeoff |
|---|---|---|
| Association/calibration | `association_max_dt_s`, `depth_scale`, intrinsics | Synchronization and metric geometry correctness |
| Dense odometry | `frame_stride`, `depth_diff_max_m` | Speed versus inter-frame alignment robustness |
| Map | `voxel_size_m`, `map_every_n_frames`, `points_per_keyframe` | Detail versus memory and compute |
| ORB retrieval | `orb_features`, `ratio_test`, `min_matches`, `min_match_score` | Recall versus false appearance candidates |
| Search policy | `min_frame_separation`, `cooldown_frames`, `max_candidates` | Redundancy/cost versus opportunity to close loops |
| PnP | RANSAC and inlier settings | Geometric proposal robustness |
| ICP | correspondence, fitness, RMSE, and correction limits | Geometric acceptance strictness |
| Graph | pruning, correspondence scale, loop preference | Influence/retention of uncertain loop edges |
| Output | `map_voxel_size_m` | Saved map size versus detail |

The configuration has no schema validator. Invalid intervals, calibration, or
thresholds may fail at runtime or silently produce poor geometry.

## 7. Recommended experiments

1. Run autonomous with and without `groundtruth.txt`. The estimated outputs should
   remain RGB-D-only; only blue visualization and accuracy fields change.
2. Compare autonomous and odometry-only map/path outputs on the same frames.
3. Tighten `ratio_test` and watch appearance candidates fall relative to database
   comparisons.
4. Raise `min_pnp_inliers`, then tighten ICP thresholds, to distinguish proposal
   rejection from registration rejection.
5. Compare `frame_stride: 1`, `3`, and `6`; inspect tracking success and loop
   opportunities rather than speed alone.
6. Compare map and output voxel sizes using `map_ms`, saved points, and extent.
7. Use GT-assisted only to demonstrate graph deformation; do not treat its
   optimized GT error as an autonomous benchmark.
8. Plot per-frame CSV latency and compare it with aggregate JSON percentiles.

## 8. Controls and practical limits

`Space` pauses/resumes while retaining viewer interaction. Left drag orbits,
`Ctrl`+left or middle drag pans, the wheel zooms, `R` resets, `H` shows Open3D
help, and `Q`/`Esc` quits. Use `--no-viewer --no-realtime` in headless sessions.

Long runs grow the graph, keyframe database, and map without eviction. CPU
processing is sequential inside the worker. There is no dynamic-object removal,
TSDF/surfel fusion, checkpoint resume, GPU selection, or automatic recovery after
tracking loss. These omissions keep the implementation small enough to inspect.

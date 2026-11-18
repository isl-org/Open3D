# RGB-D localization and pose-graph SLAM learning demo

This directory is a runnable, instrumented SLAM lesson built in a sparse fork of
Open3D. It turns synchronized RGB and depth images into dense frame-to-frame
odometry, autonomous loop constraints, a globally optimized camera trajectory,
and a colored point-cloud map. The default `autonomous` mode estimates motion and
loop closures from RGB-D alone. If TUM motion-capture ground truth is present, it
is used only for display and end-of-run evaluation in that mode.

The goal is observability and pedagogy, not benchmark or production performance.
The program exposes local drift, loop-search selectivity, geometric verification,
global correction, a deliberately simple covariance display, per-frame timing,
serialized outputs, and end-of-run KPIs.

> [!IMPORTANT]
> This is an educational CPU demo. Its dense RGB-D odometry, ORB retrieval,
> depth-backed PnP, ICP, and Open3D pose-graph optimization are real, but its
> covariance is heuristic, its map is a concatenation of keyframe clouds, and it
> has no relocalization, persistent landmarks, bundle adjustment, IMU fusion, or
> production-grade loop-management policy.

## Quick start

```bash
cd /home/jacob/airobotics/3d_slam_example
source .venv/bin/activate
python slam_demo.py
```

For a clean setup:

```bash
./setup.sh
source .venv/bin/activate
python download_dataset.py
python slam_demo.py
```

The default in `config.yaml` is `slam.mode: autonomous`. Useful invocations are:

```bash
# Explicit default: RGB-D odometry plus autonomous RGB-D loop closure.
python slam_demo.py --mode autonomous

# Sequential RGB-D odometry and mapping, with no loop edges or optimization.
python slam_demo.py --mode odometry-only

# Scheduled teaching constraints derived from groundtruth.txt.
python slam_demo.py --mode gt-assisted

# Alias for odometry-only.
python slam_demo.py --no-loop-closure

# Switch to GT-assisted mode and set its scheduled constraint interval.
python slam_demo.py --loop-closure-every 20

# Headless processing; do not pace consumption to playback_hz.
python slam_demo.py --no-viewer --no-realtime --max-frames 100
```

`--dataset`, `--config`, `--max-frames`, `--seed`, `--no-image-window`, and
`--no-realtime` provide the remaining common overrides. Run
`python slam_demo.py --help` for the complete CLI.

## Three modes

| Mode | Motion source | Loop constraints | Ground truth |
|---|---|---|---|
| `autonomous` (default) | Dense RGB-D odometry | ORB retrieval, depth-backed `solvePnPRansac`, ICP verification, then pose-graph optimization | Optional; display and KPI evaluation only, never consumed by estimation |
| `gt-assisted` | Dense RGB-D odometry | Scheduled node-0-to-current constraints derived from GT every `loop_closure_every_n_frames` | Required and consumed to construct constraints |
| `odometry-only` | Dense RGB-D odometry | None; the sequential graph is still built and saved | Optional; display and KPI evaluation only |

`--no-loop-closure` selects `odometry-only`. `--loop-closure-every N` selects
`gt-assisted` and overrides its interval, regardless of the configured mode.
The autonomous detector runs only on map keyframes and does not use
`loop_closure_every_n_frames`.

In autonomous and odometry-only modes, `groundtruth.txt` does not affect frame
association, odometry, loop detection, optimization, or mapping. When available,
associated GT poses are normalized to the first available GT pose and used for
the blue path and accuracy KPIs. In `gt-assisted`, records without associated GT
are removed before stride and frame limiting, and the run fails if none remain.

## Dataset layout

The minimum no-GT layout is:

```text
dataset/
|-- rgb.txt                 # timestamp relative/path/to/rgb.png
|-- depth.txt               # timestamp relative/path/to/depth.png
|-- rgb/
|   `-- *.png               # 8-bit color images
`-- depth/
    `-- *.png               # single-channel depth images
```

An optional TUM-format file enables display/evaluation and `gt-assisted` mode:

```text
groundtruth.txt             # timestamp tx ty tz qx qy qz qw
```

For each RGB timestamp, `load_records()` takes the nearest depth row and rejects
the RGB record only if that RGB-depth gap exceeds `association_max_dt_s`. If
`groundtruth.txt` exists, its nearest row is attached only when its gap also fits
the tolerance; failure to associate GT does not reject an otherwise valid RGB-D
record except through the later `gt-assisted` filtering step. GT is not
interpolated. Timestamp lists are assumed sorted and nonempty.

TUM depth is converted with:

```text
depth_m = depth_raw / depth_scale
```

Depth beyond `depth_trunc_m` becomes invalid. Camera width, height, and intrinsics
must match the images. A wrong depth scale changes map and translation scale; bad
intrinsics deform both odometry and geometry.

## Pipeline

### Dense local odometry

Every adjacent processed RGB-D pair is passed to Open3D
`compute_rgbd_odometry()` with `RGBDOdometryJacobianFromHybridTerm`. The solver
uses photometric and depth/geometric consistency over a coarse-to-fine pyramid.
It is dense alignment, separate from the sparse ORB pipeline used only to propose
autonomous loop closures.

Open3D returns `T_current_previous`. Camera-to-map pose is accumulated as:

```text
T_map_current = T_map_previous * inverse(T_current_previous)
```

Each graph node stores a camera-to-map pose. A successful sequential edge stores
Open3D's relative transform and 6x6 information matrix. On odometry failure, the
pose is held and an identity edge with information `0.001 * I6` is added; the
failed current image still becomes the source for the next pair.

The orange trajectory is unmodified local odometry. The red trajectory adds a
seeded Gaussian SE(3) perturbation after each successful increment to make drift
visually obvious; it never affects the graph or map. The green trajectory is the
current pose graph, and blue appears only where GT is available.

### Autonomous place recognition

On every map keyframe (`index % map_every_n_frames == 0`),
`make_keyframe()` performs the following:

1. Detect up to `orb_features` ORB features in the grayscale RGB image.
2. Sample metric depth at each rounded keypoint pixel.
3. Drop features with invalid depth or depth at/above `max_feature_depth_m`.
4. Back-project each retained pixel to a 3D point in that keyframe's camera
   coordinates using `fx`, `fy`, `cx`, and `cy`.
5. Retain the depth-backed descriptor set and the downsampled local point cloud.

`detect_loop()` ignores database keyframes closer than
`min_frame_separation` processed indices and suppresses all search during
`cooldown_frames` after an accepted loop. For each eligible old keyframe it:

1. Uses brute-force Hamming KNN matching (`k=2`) from old descriptors to current
   descriptors, without cross-checking.
2. Applies Lowe's ratio condition
   `best_distance < ratio_test * second_best_distance`.
3. Computes `match_score = good_matches / min(old_descriptor_count,
   current_descriptor_count)` with a denominator floor of one.
4. Keeps a place candidate only if both `min_matches` and `min_match_score` pass.
5. Ranks candidates by absolute good-match count and geometrically checks at most
   `max_candidates` strongest candidates.

This is a small linear keyframe database, not a vocabulary tree, bag-of-words
index, learned descriptor, or scalable retrieval system.

### Depth-backed PnP and ICP verification

For each shortlisted candidate, old-keyframe 3D points are paired with matched
current-frame 2D pixels. OpenCV `solvePnPRansac(..., SOLVEPNP_EPNP)` estimates the
candidate-camera to current-camera transform. The proposal must satisfy both
`min_pnp_inliers` and `min_pnp_inlier_ratio`; RANSAC iteration count,
reprojection threshold, and confidence come from `config.yaml`.

Point-to-point Open3D ICP then aligns the old and current local keyframe clouds,
initialized by PnP. A loop is accepted only when:

- ICP `fitness >= min_icp_fitness`;
- ICP `inlier_rmse <= max_icp_rmse_m`;
- ICP changes the PnP translation by at most
  `max_icp_translation_correction_m`; and
- ICP changes the PnP rotation by at most
  `max_icp_rotation_correction_deg`.

The correction guards help reject a superficially plausible registration on
repeated or weak geometry. The accepted ICP transform and an Open3D information
matrix computed from the aligned clouds become an uncertain edge from the old
keyframe node to the current node. Search stops after the first verified
candidate, and the current keyframe is then added to the database.

### GT-assisted constraints

`gt-assisted` mode does not run autonomous retrieval. At positive multiples of
`loop_closure_every_n_frames`, it adds an uncertain edge from node zero to the
current node:

```text
T_current_camera0 = inverse(T_map_current_gt) * T_map_camera0_gt
information       = loop_information * I6
```

These are synthetic teaching constraints. They do not prove that the camera
actually revisited node zero and must not be presented as autonomous SLAM.

### Pose-graph optimization

After either kind of accepted loop edge, `optimize_pose_graph()` runs Open3D's
Levenberg-Marquardt global optimization with node zero fixed. Sequential edges
are certain; loop edges are uncertain and are subject to Open3D's configured
edge-pruning and loop-preference behavior. The optimization uses
`max_correspondence_distance_m`, `edge_prune_threshold`, and
`preference_loop_closure` from `config.yaml`.

The code then takes the final optimized node as the live corrected pose, refreshes
the complete green trajectory, and rebuilds every cached map keyframe at its
optimized node pose. `correction_jump_m` is only the Euclidean translation change
of the current endpoint before versus after optimization; it is not the total
graph deformation or an accuracy score. The implementation reports an accepted
constraint before checking whether Open3D later pruned or effectively ignored
that uncertain edge.

This is pose-graph optimization, not bundle adjustment. Only camera poses and
relative SE(3) constraints are optimized. There are no persistent landmark IDs
or camera-landmark reprojection residuals.

### Mapping and uncertainty

Each frame is back-projected, voxel-downsampled, and randomly capped at
`points_per_keyframe`. The cyan current scan is transformed every frame. Every
`map_every_n_frames` frame is cached in camera coordinates and added to the RGB
map. A loop correction rebuilds the map from those cached local clouds.

The map is not TSDF, surfel, occupancy, or mesh fusion. It has no dynamic-object
filter, cross-keyframe deduplication, or bounded keyframe cache. The final writer
can apply one additional global voxel downsample before serialization.

The yellow rings are derived from a heuristic diagonal 6x6 covariance. Process
variance is added once per frame transition and the full matrix is multiplied by
`loop_closure_shrink` after a reported closure. It does not use SE(3) Jacobians,
the graph information matrices, correlations, or a calibrated measurement
update, so it is not a statistically valid uncertainty estimate.

## Display and controls

| Color or object | Meaning |
|---|---|
| RGB-colored points | Accumulated keyframe map in the first-camera map frame |
| Cyan points | Current scan transformed by the corrected pose |
| Magenta sphere / cyan frustum | Current corrected camera position/orientation |
| Orange path | Local dense RGB-D odometry |
| Red path | Artificially perturbed teaching trajectory |
| Green path | Final/current pose-graph node poses |
| Blue path | Optional normalized ground truth |
| Yellow rings | Heuristic positional covariance at `ellipsoid_sigma` |

The estimated map origin is the first camera pose, independently of GT. If GT
exists, its earliest associated pose is separately normalized to identity. In the
usual dataset that pose belongs to frame zero; if early frames lack GT, the two
origins are not additionally aligned.

| Input | Action |
|---|---|
| Left drag | Orbit/tilt |
| `Ctrl` + left drag or middle drag | Pan |
| Wheel | Zoom |
| `R` | Reset view |
| `H` | Open3D controls |
| `Space` | Pause/resume processing and playback while keeping GUI events active |
| `Q` in RGB-D window or `Esc` | Quit |

The image window shows the exact RGB image and metric depth used by odometry.
`playback_hz` is a synthetic consumption cadence, not dataset timestamp replay.

## Output files

At normal shutdown the worker writes its graph and the consumer writes the
accumulated state. The consumer also attempts to save its partial state after a
user exit, but an interrupted worker may not reach graph serialization. Paths
under `output/` are configurable and replace same-named files when their writers
run; `telemetry.csv` has a fixed path beside `slam_demo.py` and is opened in
write mode at worker startup.

| File | Contents |
|---|---|
| `telemetry.csv` | One flushed row per worker-produced frame: timings, tracking status, loop-search counts, accepted-loop details, correction jump, and heuristic position uncertainty |
| `output/autonomous_map.ply` | Compressed colored point cloud corresponding to the final displayed map, after optional `output.map_voxel_size_m` downsampling |
| `output/trajectories.npz` | XYZ position arrays named `noisy`, `local_odometry`, `optimized`, and `ground_truth`; despite the filename, these are positions, not 4x4 poses |
| `output/pose_graph.json` | Open3D pose graph containing all processed nodes, certain sequential edges, and accepted uncertain loop edges |
| `output/run_summary.json` | Structured end-of-run KPIs described below |

The GT trajectory array can be empty and contains only frames that had associated
GT; it does not carry frame indices. The pose graph is written by the worker when
its loop exits. The map, trajectories, and summary are written by the consumer
after shutdown. If no frame state is consumed, no run summary is produced.

## KPIs

At shutdown, the CLI prints `=== Run KPI Summary ===` and writes the complete
machine-readable report to `output/run_summary.json` (or
`output.run_summary_file`). The report describes only frame states consumed by
the main thread. Accuracy is present only if at least one of those frames has an
associated GT pose.

### Run and tracking

| JSON field | Definition and interpretation |
|---|---|
| `run.mode` | Effective mode after CLI overrides |
| `run.frames_processed` | Number of consumed frame samples, including frame zero |
| `run.elapsed_s` | Wall time from consumer startup through processing, pacing/pauses, GUI handling, shutdown, and map/trajectory serialization; initialization before `consume()` and summary writing are outside it |
| `run.ground_truth_available` | Whether any consumed frame had associated GT |
| `tracking.transitions_attempted` | `max(frames_processed - 1, 0)`; frame zero has no pairwise transition |
| `tracking.successful_transitions` | Count of attempted transitions for which Open3D returned `odometry_ok=True` |
| `tracking.success_rate_percent` | `100 * successful / attempted`, or `100` when no transition was attempted |

Tracking success is Open3D's pairwise solver status, not proof of an accurate
pose. A visually or numerically poor transform can still be reported successful.

### Performance

For each worker stage `io`, `odometry`, `map`, and `total`,
`performance.latency.<stage>` contains `mean_ms`, `median_ms`, and `p95_ms` over
consumed samples. `total` measures worker computation from the per-frame timer
through map preparation. It includes autonomous retrieval/verification and graph
optimization when they happen, but those costs are not broken into separate
columns. It excludes queue wait, playback sleep, consumer bookkeeping, geometry
upload, and rendering. Frame-zero `io_ms` and `odometry_ms` are explicitly zero,
and its initial RGB-D read happened before that frame's timer.

| JSON field | Formula / limitation |
|---|---|
| `performance.effective_throughput_hz` | `frames_processed / run.elapsed_s`; includes non-compute wall costs and final map/trajectory saving, so it is the run-level observed rate |
| `performance.compute_throughput_hz` | `1000 / mean(total_ms)`; an inverse mean worker cost, not sustained end-to-end throughput |
| `performance.latency.*.mean_ms` | Arithmetic mean |
| `performance.latency.*.median_ms` | 50th percentile |
| `performance.latency.*.p95_ms` | NumPy 95th percentile of available frame samples; small runs give a weak tail estimate |

### Loop closure

| JSON field | Definition and availability |
|---|---|
| `loop_closure.accepted` | Frames marked as closures. Autonomous: geometrically verified loop constraints. GT-assisted: scheduled GT-derived constraints. Odometry-only: zero |
| `loop_closure.database_comparisons` | Sum of eligible old/current keyframe comparisons in autonomous search; zero in other modes and during cooldown/non-keyframes |
| `loop_closure.appearance_candidates` | Sum of candidates passing both ORB match thresholds before `max_candidates` truncation |
| `loop_closure.geometric_checks` | Number of shortlisted candidates actually passed to PnP/ICP; search stops at the first acceptance |
| `loop_closure.autonomous_acceptance_rate_percent` | `100 * accepted autonomous loops / geometric_checks`; JSON `null` when no geometric check ran. GT-assisted closures are excluded |
| `loop_closure.mean_matches` | Mean ORB good-match count over accepted autonomous loops; JSON `null` when none were accepted |
| `loop_closure.mean_pnp_inliers` | Mean RANSAC PnP inlier count over accepted autonomous loops; JSON `null` when none were accepted |
| `loop_closure.mean_pnp_inlier_ratio` | Mean of `pnp_inliers / loop_matches` over accepted autonomous loops; JSON `null` when none were accepted |
| `loop_closure.mean_icp_fitness` | Mean Open3D ICP overlap fitness over accepted autonomous loops; JSON `null` when none were accepted |
| `loop_closure.mean_icp_rmse_m` | Mean ICP inlier RMSE over accepted autonomous loops; JSON `null` when none were accepted |
| `loop_closure.mean_correction_m` | Mean endpoint translation jump over accepted closure frames; zero if none |
| `loop_closure.max_correction_m` | Maximum endpoint translation jump over accepted closure frames; zero if none |

These are event and selectivity metrics, not precision/recall. There is no
independent loop-closure label set, and accepted GT-assisted events are not
comparable to autonomous detections. Correction magnitude can be small for a
useful loop or large for a bad one and does not measure residual reduction.

### Trajectory and map

| JSON field | Definition / limitation |
|---|---|
| `trajectory.local_path_length_m` | Sum of Euclidean distances between consecutive local-odometry positions |
| `trajectory.optimized_path_length_m` | Same sum over final pose-graph node positions |
| `map.saved_points` | Point count after the final optional global voxel downsample; zero if there was no map to write |
| `map.extent_m` | Axis-aligned `[max_x-min_x, max_y-min_y, max_z-min_z]` of the saved cloud, or `[0,0,0]` when empty |

Path length ignores orientation and is not distance to GT. A shorter optimized
path is not inherently more accurate. Map point count and extent depend on depth
range, keyframe interval, random point capping, voxel sizes, tracking, and early
termination; they are not map-quality scores.

### Accuracy when GT is available

`accuracy.local_odometry` and `accuracy.optimized` each contain:

| JSON field | Formula / interpretation |
|---|---|
| `associated_frames` | Number of pose indices with associated GT |
| `ate_rmse_m` | `sqrt(mean(||t_est[i] - t_gt[i]||^2))` |
| `ate_median_m` | Median absolute translation error |
| `ate_max_m` | Maximum absolute translation error |
| `rpe_translation_rmse_m` | RMSE of translation magnitude in `inverse(delta_gt) * delta_est` for consecutive GT-associated indices |
| `rpe_rotation_rmse_deg` | RMSE of rotation-angle magnitude from that same relative error transform |
| `final_position_error_m` | Absolute translation error at the last GT-associated index |

The estimate anchors graph node zero at identity; GT is normalized by its earliest
available associated pose. No later SE(3) or Sim(3) alignment is fitted before
ATE. RPE pairs consecutive available GT-associated indices, which may span more
than one processed transition when GT is missing. With only one associated
frame, both RPE fields are JSON `null`.
Accuracy is evaluation-only in autonomous and odometry-only modes. In
`gt-assisted`, optimized accuracy is not independent because GT generated the
loop constraints.

### CLI summary versus persisted reports

The final CLI summary intentionally displays only a subset: mode, frames,
elapsed time, tracking count/rate, mean and p95 total latency, compute throughput,
accepted loops/geometric checks/database comparisons, correction mean/max, local
and optimized path lengths, saved map count/extent, autonomous acceptance/PnP/ICP
quality when available, and ATE RMSE plus translation and rotation RPE RMSE when
GT exists. Fields such as effective throughput, latency medians and per-stage
distributions, appearance candidates, mean match/inlier counts, ATE median and
maximum, associated-frame count, and final position error remain available in
`run_summary.json`.

`telemetry.csv` is the per-frame diagnostic record. Its columns are:

| CSV column | Meaning |
|---|---|
| `frame`, `timestamp` | Processed index and original RGB timestamp |
| `io_ms`, `odometry_ms`, `map_ms`, `total_ms`, `processing_hz` | Worker stage timings and `1000 / total_ms` |
| `odometry_ok` | Pairwise solver success flag |
| `closure`, `loop_source` | Accepted closure flag and source graph node, blank when none |
| `loop_compared` | Eligible database keyframes compared on this frame |
| `loop_appearance_candidates` | Candidates passing ORB thresholds on this frame |
| `loop_geometric_checks` | Candidates submitted to PnP/ICP on this frame |
| `loop_matches`, `pnp_inliers` | Match and PnP-inlier counts for an accepted autonomous loop, otherwise zero |
| `icp_fitness`, `icp_rmse` | Accepted autonomous loop ICP quality, otherwise blank |
| `correction_jump_m` | Current endpoint translation change after optimization |
| `position_std_m` | Square root of the largest eigenvalue of the heuristic positional covariance block |

The live one-line telemetry additionally shows consumer cadence (`viewer Hz`) and
worker-state-to-pre-render delay (`e2e`). These are not persisted. The CSV does
not contain aggregate KPIs, final accuracy, path/map statistics, or wall-clock
effective throughput; `run_summary.json` does. Conversely, the JSON does not
retain accepted-loop match/inlier/ICP details or individual frame timings; the
CSV does. Use both files for run analysis.

## Configuration reference

All distances are meters and angles are degrees unless named otherwise.

| Group | Important settings |
|---|---|
| `dataset` | Path, depth scale/truncation, timestamp tolerance |
| `camera` | Image dimensions and pinhole intrinsics |
| `slam.mode` | `autonomous`, `gt-assisted`, or `odometry-only` |
| `processing` | Stride/frame limit, cloud voxel/cap/keyframe interval, odometry depth threshold, GT interval, playback rate |
| `autonomous_loop` | ORB count/depth, separation/cooldown, ratio and appearance thresholds, candidate cap, PnP RANSAC thresholds, ICP acceptance and correction guards |
| `optimization` | GT edge information and Open3D global-optimization options |
| `noise` | Artificial red-trajectory perturbation only |
| `covariance` | Heuristic initial/process variances, closure shrink, display sigma |
| `visualization` | Window and geometry display sizes |
| `output` | Output directory/names and final map voxel size |

The YAML has no schema or range validation. `max_frames`, stride, intervals, and
thresholds should be chosen carefully. `--max-frames 0` falls back to the YAML
value because the CLI uses Python `or`, and a negative maximum uses Python's
negative slicing semantics.

## Architecture and files

The worker thread owns I/O, odometry, autonomous detection, graph updates,
optimization, map preparation, covariance, and CSV writing. A bounded queue of
two `FrameState` objects applies backpressure. The main thread owns Open3D/OpenCV
events, pacing, display arrays, live terminal output, and final map/trajectory/KPI
serialization.

| Path | Responsibility |
|---|---|
| `slam_demo.py` | Executable pipeline, visualization, persistence, and KPIs |
| `autonomous_loop.py` | ORB keyframes/retrieval, PnP estimation, and ICP verification |
| `config.yaml` | Active defaults and thresholds |
| `SLAM_GUIDE.md` | Shorter conceptual walkthrough and experiments |
| `PLAN.md` | Locked implementation/documentation scope and evidence |
| `download_dataset.py` | TUM sequence downloader/extractor |
| `examples/python/pipelines/rgbd_odometry.py` | Retained upstream odometry reference |
| `examples/python/reconstruction_system/` | Retained upstream reconstruction/pose-graph references |

## Experiments

1. Compare `--mode autonomous` and `--mode odometry-only`; inspect accepted
   loops, path lengths, and the final map rather than assuming every correction
   improves accuracy.
2. Temporarily move `groundtruth.txt` aside and run autonomous mode to verify the
   complete RGB-D-only path; accuracy fields and the blue trajectory disappear.
3. Change `ratio_test`, `min_matches`, and `min_match_score`; compare database
   comparisons, appearance candidates, geometric checks, and accepted loops.
4. Tighten PnP and ICP thresholds separately to see where candidate rejection
   moves from appearance retrieval to geometry.
5. Compare `frame_stride` values. Larger baselines reduce work but can damage
   dense odometry and sparse loop matching.
6. Compare `voxel_size_m`, `map_every_n_frames`, and
   `output.map_voxel_size_m`; inspect per-frame `map_ms`, saved point count, and
   extent.
7. Run `--mode gt-assisted` only as a teaching comparison. Its optimized GT
   metrics are circular because GT supplies its global constraints.
8. Set artificial noise to zero to make red coincide with orange; verify that
   green and serialized map outputs are unaffected.

## Limitations

- ORB retrieval is a brute-force scan over eligible keyframes and is vulnerable
  to perceptual aliasing, texture scarcity, illumination changes, and missing
  feature depth.
- PnP and ICP thresholding reduces false positives but cannot prove loop
  correctness; there is no switchable constraint, robust multi-hypothesis loop
  manager, or loop precision/recall benchmark.
- There are no persistent landmarks, classical bundle adjustment, IMU fusion,
  relocalization after tracking loss, motion-model initialization, or backend
  marginalization.
- Mapping is a growing keyframe point-cloud union, not a fused or bounded map.
- The covariance display is pedagogical and must not be interpreted as calibrated
  state uncertainty.
- The implementation uses Open3D legacy CPU interfaces and does not select CUDA
  or SYCL devices.
- Output files are run artifacts, not resumable checkpoints.

## Troubleshooting

Use `--no-viewer --no-realtime` over SSH or without a working OpenGL display.
On a Wayland desktop with XWayland available, the launcher automatically selects
the X11 compatibility backend because Open3D's legacy GLFW viewer can fail GLEW
initialization on native Wayland. `--no-image-window` disables only the RGB-D
OpenCV window; use `--no-viewer` to disable both GUI windows.
Reduce `frame_stride` if odometry fails. Verify calibration and `depth_scale` if
the map bends or has the wrong scale. Increase voxel sizes, keyframe spacing, or
reduce `points_per_keyframe` if mapping/rendering is slow. Autonomous runs with
no accepted loop are still valid; inspect retrieval and verification counts
before loosening thresholds.

## Provenance

This is a sparse clone of [`isl-org/Open3D`](https://github.com/isl-org/Open3D)
at upstream commit `b6c5e196`. The demo follows the retained upstream RGB-D
odometry and reconstruction examples. The default data is the TUM RGB-D
`rgbd_dataset_freiburg1_xyz` sequence from Sturm et al., *A Benchmark for the
Evaluation of RGB-D SLAM Systems*, IROS 2012.

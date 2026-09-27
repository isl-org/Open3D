# Implementation lock

## Locked goal

Provide a runnable, inspectable RGB-D SLAM lesson in the sparse Open3D fork. The
default system must estimate local motion, discover and geometrically verify loop
closures, optimize a pose graph, rebuild and serialize a point-cloud map, and
report useful run KPIs without requiring ground truth.

## Locked requirements

- Keep the upstream Open3D examples unchanged and implement the lesson in the
  root-level demo files.
- Make `autonomous` the default mode. Its estimation path must consume RGB-D
  only; optional `groundtruth.txt` is display/evaluation data, not an estimator
  input.
- Retain explicit `gt-assisted` teaching mode and `odometry-only` baseline mode.
- Use dense Open3D RGB-D odometry for sequential motion.
- For autonomous loops, use depth-backed ORB keyframes, Hamming KNN ratio-match
  retrieval, absolute and normalized appearance thresholds, RANSAC EPNP,
  point-to-point ICP, fitness/RMSE checks, and limits on ICP correction from PnP.
- Insert verified loop transforms as uncertain graph edges, run Open3D
  Levenberg-Marquardt pose-graph optimization, and rebuild cached keyframe clouds
  using optimized node poses.
- Preserve the pedagogical noisy trajectory and heuristic covariance while
  labeling both accurately.
- Persist per-frame telemetry, final map, position trajectories, Open3D pose
  graph, and a machine-readable end-of-run KPI summary.
- Support datasets containing only `rgb.txt`, `depth.txt`, RGB images, and depth
  images. Require GT only in `gt-assisted` mode.
- Keep limitations explicit: no production loop manager, persistent landmarks,
  bundle adjustment, rigorous covariance propagation, fused/bounded map,
  relocalization, or IMU fusion.

## Locked validation

- Compile the Python entry points.
- Exercise RGB/depth association with and without `groundtruth.txt`.
- Run at least ten real TUM frames headlessly.
- Exercise all three mode selections and verify GT is required only by
  `gt-assisted`.
- Unit-check autonomous retrieval/verification acceptance and rejection paths
  with deterministic fixtures where practical.
- Verify that `telemetry.csv`, map PLY, trajectory NPZ, pose-graph JSON, and run
  summary JSON are produced and readable.
- Cross-check every documented field, formula, mode, and output against
  `slam_demo.py`, `autonomous_loop.py`, and `config.yaml`.

## Risks and mitigations

- Desktop OpenGL may be unavailable: retain `--no-viewer` and headless tests.
- Dense odometry can fail on large baselines or blurred frames: hold the pose,
  add a weak identity edge, and report failure rather than substituting GT.
- Appearance aliasing can propose false loops: require depth-backed PnP and ICP,
  use fitness/RMSE thresholds, and bound ICP departure from PnP.
- Sparse texture or missing feature depth can suppress valid loops: expose
  comparisons, candidates, geometric checks, matches, inliers, and ICP quality.
- Global correction can distort a point-cloud union: cache clouds in camera
  coordinates and rebuild them from optimized node poses.
- Aggregate metrics can be over-interpreted: distinguish compute from wall
  throughput, detector counts from loop accuracy, heuristic uncertainty from
  covariance, and GT-assisted accuracy from independent evaluation.
- Output growth is unbounded on long runs: document the educational scope and
  configurable voxel/keyframe/point caps rather than implying production scale.

## Status log

| Scope | Status | Decision/evidence |
|---|---|---|
| Dense RGB-D odometry and graph | Implemented | `slam_demo.py` adds one node per frame and certain sequential edges from `compute_rgbd_odometry()` |
| Autonomous mode default | Implemented | `config.yaml` sets `slam.mode: autonomous`; CLI exposes all three modes |
| Optional GT | Implemented | `load_records()` conditionally reads GT; only `gt-assisted` filters/requires associated poses |
| ORB retrieval | Implemented | `make_keyframe()` retains depth-backed ORB descriptors; `detect_loop()` applies Hamming KNN ratio matching and two appearance thresholds |
| Geometric verification | Implemented | `_verify_candidate()` applies RANSAC EPNP followed by point-to-point ICP and all configured acceptance guards |
| Global correction | Implemented | Accepted autonomous or scheduled GT edges trigger Open3D LM optimization and keyframe-map rebuild |
| Serialization | Implemented | Map PLY, trajectory NPZ, pose-graph JSON, and per-frame CSV writers are present |
| End-of-run KPIs | Implemented | `build_run_summary()` aggregates run, tracking, performance, loop, trajectory, map, and optional accuracy metrics; `print_and_save_summary()` prints a subset and writes JSON |
| Expanded documentation | Complete | README and guide now describe autonomous/default behavior, no-GT layout, all modes, loop stages, graph optimization, outputs, KPI formulas/availability, and pedagogical limits |

## Documentation evidence

- `README.md` contains the exact `## KPIs` section and distinguishes live CLI,
  per-frame `telemetry.csv`, and aggregate `output/run_summary.json` coverage.
- Every JSON metric emitted by `build_run_summary()` is documented, including
  formulas, zero/null behavior, GT availability, and interpretation limits.
- Every CSV field in `processing_loop()` is documented, including autonomous
  retrieval and verification details not aggregated into JSON.
- Output documentation matches `save_outputs()`, `write_pose_graph()`, and
  `print_and_save_summary()`: serialized maps and graphs now exist, and NPZ
  trajectories are XYZ arrays rather than full pose matrices.
- The no-GT dataset description matches conditional GT loading and clarifies the
  different filtering behavior of `gt-assisted`.
- Autonomous loop documentation follows the exact implementation order: depth
  filtering, ORB ratio retrieval, appearance ranking, RANSAC PnP, ICP checks,
  uncertain edge insertion, optimization, and map rebuild.

## Validation status

Validation completed on the bundled TUM sequence and a linked copy with no
`groundtruth.txt`: Python compilation passed; all three modes ran headlessly;
autonomous no-GT processing accepted a real loop from node 0 to node 30 in a
35-frame run; an identical-frame fixture exercised autonomous acceptance and
cooldown rejection; deterministic poses checked the ATE/RPE formulas; and the
Open3D viewer path ran for two frames. The generated JSON, NPZ, PLY, and pose
graph were reloaded and checked for expected mode, dimensions, point/node/edge
counts, autonomous search activity, and accepted-loop reporting. `git diff
--check` also passed.

# SYCL CI Timeout Investigation

## Goal

Keep the shared-library SYCL build-and-test job within GitHub Actions' six-hour
job limit while continuing to run all four GoogleTest shards.

## Findings

- The failing shared-library job used four GoogleTest shards, but GNU Parallel's
  `-k` option printed completed shard output in shard-index order. At the
  six-hour cancellation point, only shards one and two had completed and their
  buffered output was visible.
- The job had four logical CPUs available. This did not limit GNU Parallel,
  which was configured with `--jobs 4` and started four test processes.
- `docker save` followed by single-threaded `xz` compression produced a 15 GB
  image artifact and consumed about 2 hours 34 minutes before testing began.
  The C++ test step consequently began with only about 90 minutes remaining.

## Requirements

- Preserve the existing shared-library artifact format and uploads.
- Preserve four SYCL GoogleTest shards.
- Avoid changing public build or test interfaces.

## Plan

1. Use all available CPU threads for the workflow's `xz` compression.
2. Validate that the installed `xz` supports automatic thread selection and
   YAML parsing succeeds.
3. Re-run the affected workflow and confirm all four shard-completion summaries
   occur before the job's six-hour limit.

## Progress

- 2026-08-11: Implemented `xz -T 0` for the shared-library artifact.
- 2026-08-11: Confirmed local xz 5.2.5 accepts `--threads=0`; YAML parsing and
  workspace diagnostics pass.
- 2026-08-11: Replaced xz with `zstd -T0 --fast=1` and configured
  `upload-artifact` with `compression-level: 0`. The prior 15.3 GB artifact was
  accepted by GitHub, so minimizing end-to-end transfer time is preferable to
  maximizing compression ratio.
- Pending: Verify the next GitHub Actions run completes all four shards and
  compare the zstd compression, upload, download, and decompression timings.

## Decision Log

- 2026-08-11: Prefer parallel compression over reducing test concurrency. The
  timeout is dominated by artifact compression, and reducing shards would make
  the C++ test phase longer.
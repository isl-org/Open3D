Open3D is a cross-platform (Windows, Linux, macOS and x86_64, arm64, CUDA and
oneAPI platforms) C++17 and Python library for 3D data processing,
visualization, reconstruction, registration, and ML-related workflows. Other
functionality is out of scope.

## Coding and PR review directions

- For any change, first research existing code and prepare a plan with definite goal, requirements, implementation steps, test method, risks and mitigations. **Lock the goal, requirements and test method, and keep the implementation steps with a log of status and decisions made.** Record evidence and revise the lock explicitly if it changes; never expand scope silently. Add the plan summary to the PR description.
- Store intermediate live research and progress reports with important findings in .md files and refer to them as needed.
- Use subagents for major steps. 
- Create new git branches when you explore implementation options - go back to previous code as required. 
- Preserve pre-existing worktree changes. Never reset, overwrite, stash, stage, or commit unrelated work implicitly. Never rewrite published history or force-push.
- Prioritize human readability and maintainability.
- **AVOID REDUNDANT IMPLEMENTATION** when similar functionality already exists - refactor and reuse instead.
- Keep changes focused and small. Avoid broad refactors unless requested.
- Read the relevant C++ / Python / docs files together and identify whether bindings, docs, and tests must change with the source change.
- **Debugging:** List candidate root-cause hypotheses, then classify each as easy or hard to fix (edit size, blast radius, build/test cost). Test easy candidate fixes directly, cheapest first, and keep the one that resolves the issue. Before attempting a hard or complex fix, add logging or instrumentation to confirm the hypothesis. In all cases prefer the shortest path to a validated resolution: run the cheapest check that could disprove the current hypothesis, validate afterward, remove probes, and undo failed fixes.
- Developer docs: Document the code with brief (2-3 lines) comments (Why and What is the code doing?), typically for each function and file. Ensure code and docs are consistent.
- User docs: Update `docs`, Doxygen docs in C++ headers and Google Sphinx RST docs in Python bindings for new / changed code behavior.  Add / update an example function use snippet in the docs.
- For new functionality, docs and examples, prefer Tensor implementations that work on CPU+CUDA+SYCL. Do not add silent device or backend fallbacks. Do not move large data between devices without explicit user instructions.
- Use the Eigen library for Math operations and oneAPI TBB for multithreading. Avoid: OpenMP, stdgpu.
- Add unit tests for new behavior and bug fixes (if needed).  Add or update both C++ and Python tests when the feature is exposed in both layers.
- Keep dependencies minimal. Reuse functionality from existing dependencies if needed. Do not update 3rdparty code with patches.
- Preserve existing public APIs unless the task explicitly requires API changes.
- Avoid unnecessary compiler warnings, build breakages, and unrelated cleanup.
- After the first substantive edit, run the cheapest behavior-scoped check that can falsify the current hypothesis before making adjacent edits.
- Ensure the smallest set of relevant tests runs correctly locally. Report unavailable hardware or configurations and never claim unrun checks passed. The full CI must pass for all supported software and hardware platforms in GitHub.
- Before committing, inspect the final diff and status. Exclude unrelated changes, generated artifacts, temporary instrumentation, agent markers, and formatter churn.
- Use an imperative, specific commit subject and brief description and reasons for each significant change in the body. Do not amend unrelated commits or add co-author trailers unless requested.
- Push without force, setting upstream for a new branch. When creating a PR, use `.github/pull_request_template.md` and include the plan summary and test evidence.

## Repository map

- Core algorithms/data structures: `cpp/open3d/` 
- New CPU+CUDA+SYCL (Tensor based) implementations: `cpp/open3d/{core,t}`
- Legacy CPU only (Eigen based) data structures and algorithms: `cpp/open3d/{camera,geometry,io,pipelines}`
- GUI visualization (based on GLFW + Dear ImGUI + Filament renderer): `cpp/open3d/visualization/{app,gui,rendering,visualizer}`
- Legacy OpenGL + GLFW visualization: `cpp/open3d/visualization/visualizer`
- C++ tests: `cpp/tests/`
- Python bindings: `cpp/pybind/`
- Python tests: `python/test/`
- Examples: `examples/`
- Build logic: `CMakeLists.txt`, `3rdparty/find_dependencies.txt`
- Tooling/style/CI helpers: `util/`
- Docs: `docs/` and inline docs in C++ headers (Doxygen) and Python bindings.

## Read computer local environment information from AGENTS.local.md

## Build

- `mkdir -p build && cd build`
- Configure: `cmake -S .. -B . -D OPTION=VALUE...`
- Build: `cmake --build . --parallel $NPROC --target TARGET` (NPROC = numn CPUs) 
- Install: `cmake --build . --target install`

If Python bindings are required, use an activated virtual environment or set
`-DPython3_ROOT=/path/to/python`.  

### Useful CMake options (specify `=ON` or `=OFF`):

- `BUILD_UNIT_TESTS`
- `BUILD_PYTHON_MODULE`
- `BUILD_[CUDA|SYCL|ISPC]_MODULE`
- `BUILD_EXAMPLES`
- `BUILD_SHARED_LIBS`

### Useful targets:
- Python package targets: `python-package|pip-package|install-pip-package`
- C++ tests: `tests`
- C++ Binary library package: `package`
- Viewer: `Open3DViewer`
- Examples: By example name.

## Test

### C++

- `./bin/tests --gtest_list_tests`
- `./bin/tests --gtest_filter=*TEST_NAME*`

### Python

- `pip install -r python/requirements_test.txt`
- `cmake --build build --target install-pip-package`
- `pytest ../python/test/SPECIFIC/TEST_FILE.PY`

## Style and formatting

- C++ follows Google-style conventions with Open3D-specific rules.
- Use 4 spaces, no tabs.
- Use `#pragma once` for header guards.
- Prefer smart pointers over naked pointers.
- Use C++17, not C++20+.
- Python follows Google Python style.

Use the repository's own formatting tools and pinned versions.

- `pip install -r python/requirements_style.txt`
- `python util/check_style.py --apply --changed-only` or `cmake --build build --target apply-style --parallel`
- Use gh for authenticated access to github APIs.
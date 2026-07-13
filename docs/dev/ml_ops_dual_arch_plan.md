# Restore dual-arch (CPU+CUDA) ML ops in the CUDA wheel

## Goal (locked)
An Open3D **CUDA** wheel (`open3d-*.whl`) must contain **both** the CPU-linked and
CUDA-linked PyTorch/TensorFlow ops, and the Python loader must pick the correct one at
runtime based on whether the user's Torch/TF has CUDA. This restores the behavior that
existed **before** the ss/dll merge (`4eb88654e`, "simplified python package", #7516) while
**keeping** ss/dll's single flat `open3d.pybind` module + single flat `libOpen3D`.

## Requirements (locked)
- CUDA wheel bundles `open3d/cuda/open3d_torch_ops.*` (CUDA torch) AND
  `open3d/cpu/open3d_torch_ops.*` (CPU torch); same for `open3d_tf_ops`.
- Loader order: try `cuda` (only if `BUILD_CUDA_MODULE` and installed torch has matching
  CUDA), then `cpu`. CPU-only wheel keeps just `cpu/`.
- `pybind*` and `libOpen3D` stay FLAT in `open3d/` (do NOT revert the single-DLL design).
- CPU-only `open3d-cpu` wheel unchanged (only `cpu/` ops).

## Pre-ss/dll mechanism (reference: `git show 4eb88654e^1:...`)
- Loaders `python/open3d/ml/torch/__init__.py` + `ml/tf/python/ops/lib.py`:
  `_lib_arch = ('cuda','cpu')`/`('cpu',)`, path `<root>/<arch>/open3d_*_ops<ext>`.
- Ops CMake `cpp/open3d/ml/{pytorch,tensorflow}/CMakeLists.txt`: output dir
  `${OPS_DIR}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>`.
- Packaging `cpp/pybind/make_python_package.cmake`: for each compiled module, scan
  `base_dir/{cpu,cuda}` and copy present arch dirs into `open3d/{cpu,cuda}`.
- `util/ci_utils.sh build_pip_package`: single build dir — build CPU (ops->cpu/), keep the
  cpu ops, reinstall CUDA torch, rebuild CUDA (ops->cuda/) WITHOUT wiping cpu/, then package
  once so both arch dirs are bundled.

## Design decision
- Only the **ops** get `cpu/`/`cuda/` subdirs again; pybind/libOpen3D remain flat.
- CPU ops (CPU torch) are dlopen'd alongside the flat CUDA `libOpen3D`; Open3D's public C++
  ABI is arch-independent and the soname is identical, so resolution is safe (validated by
  CI). Ops do not reference CUDA-only Open3D symbols.

## Implementation steps
1. Loaders: restore `_lib_arch` selection in `ml/torch/__init__.py` and `ml/tf/.../lib.py`.
2. Ops CMake: restore arch subdir output in `pytorch/` and `tensorflow/CMakeLists.txt`.
3. `make_python_package.cmake`: copy ops from `base/{cpu,cuda}` -> `open3d/{cpu,cuda}`;
   keep flat copy for pybind/libOpen3D (hybrid, detect arch parent dir).
4. `ci_utils.sh`:
   - `build_pip_package_from_installed` (Ubuntu CI): after building the CPU wheel, copy its
     `cpu/` ops into the CUDA build tree's ops `cpu/` dir before `make pip-package`, so the
     CUDA wheel bundles both.
   - `build_pip_package` (local/other): restore old two-pass build in one dir.
5. Revert the temporary "skip ml_ops" workarounds in `test_wheel`/`run_python_tests` once
   the CUDA wheel can load CPU ops with CPU torch.

## Test method (locked)
- Local: parse-check CMake, byte-compile Python loaders, simulate `make_python_package`
  arch copy with mock dirs, `bash -n util/ci_utils.sh`, `actionlint`.
- CI: Ubuntu Wheel `open3d-*.whl` on CPU-only runner must import `open3d.ml.torch` (loads
  `cpu/` ops) AND pass GPU CI (loads `cuda/`). `open3d-cpu` unchanged.

## ABI note (verified)
`BUILD_CUDA_MODULE` is a PRIVATE compile definition (Open3DSetGlobalProperties.cmake), applied
per target. The CPU ops (built without it) dlopen the CUDA `libOpen3D` in the CUDA wheel. This
is safe: the conditionals in public core headers only ADD CUDA-only entities (e.g.
`MemoryManagerCUDA`, extra CUDA free-function decls) between complete classes; they do NOT
change the layout of the shared value types exchanged with the ops (`Tensor`, `Device`,
`Dtype`, `Blob` have no such conditionals). Unlike pre-#7516 (static, self-contained ops), the
shared-lib design mixes a CPU-view ops lib with a CUDA-view libOpen3D, but only via the stable
public ABI.

## Risks / mitigations
- Cross-arch dlopen of CPU ops against CUDA libOpen3D -> mitigations: stable public ABI (see
  ABI note), identical soname; covered by CPU-runner import + ml_ops tests on the CUDA wheel.
- Windows builds no CUDA wheel -> ops stay `cpu/` only; loader falls to `cpu`. No change.
- Wheel size grows (two ops copies) -> acceptable, matches pre-ss/dll.

# Plan: Address Copilot Workflow Reviews

## Goal
Address Copilot workflow reviews for `ubuntu.yml` and `windows.yml` to make them more robust and maintainable.

## Requirements
1. **`ubuntu.yml`**:
   - Use `matrix.BUILD_SHARED_LIBS` consistently in `if:` expressions instead of `env.BUILD_SHARED_LIBS` to ensure attestation and artifact steps execute correctly.
2. **`windows.yml`**:
   - Avoid hard-coded paths like `C:/Open3D/build/CMakeCache.txt` and `C:/Program Files/Open3D/...`.
   - Track/define installation path as `INSTALL_DIR` at the workflow environment level.
   - Construct `CMakeCache.txt` path and Open3D application binary path using `$env:BUILD_DIR`, `$env:INSTALL_DIR`, and PowerShell `Join-Path`.

## Implementation Steps
- [x] Create `workflow_fixes_plan.md` to document research and decisions.
- [x] Modify `.github/workflows/ubuntu.yml` to replace `env.BUILD_SHARED_LIBS` with `matrix.BUILD_SHARED_LIBS` in `if:` expressions of the attestation/upload steps.
- [x] Modify `.github/workflows/windows.yml` to define `INSTALL_DIR: "C:\\Program Files\\Open3D"` in the workflow environment.
- [x] Replace hard-coded `"C:/Open3D/build/CMakeCache.txt"` in `windows.yml` with a path joined dynamically via `Join-Path`.
- [x] Replace hard-coded `"C:/Program Files/Open3D"` references with the `INSTALL_DIR` environment variable.
- [ ] Run validation.

## Test Method
Since we are only modifying CI/CD workflows and cannot run GitHub Actions locally, we will:
1. Verify syntax of the modified YAML files.
2. Ensure that GHA syntax is correct.

## Risks and Mitigations
- **Risk**: Syntactical errors in GitHub Actions workflow YAML.
- **Mitigation**: Double-check indentation, quotes, and GHA expression syntax before committing.

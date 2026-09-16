# Open3D release process

## Generate release assets

After the version bump has merged to `main`, authenticate with GitHub and
dispatch the release workflows with developer builds disabled:

```bash
gh auth login

for workflow in \
    ubuntu.yml \
    ubuntu-wheel.yml \
    ubuntu-cuda.yml \
    ubuntu-sycl.yml \
    ubuntu-openblas.yml \
    macos.yml \
    windows.yml; do
    gh workflow run "$workflow" --repo isl-org/Open3D --ref main \
        --field developer_build=OFF
done
```

Monitor the dispatched runs and download their artifacts with:

```bash
gh run list --repo isl-org/Open3D --branch main --limit 20
gh run watch RUN_ID --repo isl-org/Open3D --exit-status
gh run download RUN_ID --repo isl-org/Open3D
```
Upload the collected assets to the existing draft release. Do not publish unsigned apps (macOS and Windows viewers) 

```bash
draft_ref="untagged-5fff9c34536d727e8591" # from the release webpage URL
find $RELEASE_DIR -type f -exec \
gh release upload "${draft_ref}" {}  --repo "${REPO}"  --clobber \;
```

Upload wheels to PyPI:

```bash
export TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<TOKEN>  TWINE_REPOSITORY_URL="https://upload.pypi.org/legacy/"
find $RELEASE_DIR -type f -name '*.whl' -exec twine check {}  \;
find $RELEASE_DIR -type f -name '*.whl' -exec twine upload --skip-existing {}  \;
```

## Release checklist

Collect all release artifacts in the [Github draft release page](https://github.com/isl-org/Open3D/releases)

-   [ ] Create release notes draft
-   [ ] Create release videos
-   [ ] Version bump PR, merge to main
-   [ ] Build all Python version wheels (Linux, macOS, Windows x86_64) (CI)
-   [ ] Build Open3D C++ libraries (Linux, macOS, Windows x86_64)  (CI)
-   [ ] Build docs (CI) and push to [Open3d_website repo](https://github.com/isl-org/Open3D_website)
-   [ ] Build Open3D app (deb, msix)
    -   [ ] Ubuntu 20.04, Windows 10, macOS 10.15 x86_64: (CI)
    -   [ ] macOS (arm64) sign (desktop):

        From build/bin directory: `../../cpp/apps/sign_open3d_app.sh Open3D.app ../../cpp/apps/Open3DViewer/Open3dViewer.entitlements <apple-id>
                                    <cert-name> <team-id> <app-password>`

    -   [ ] Windows app sign, uploade to Windows store.
-   [ ] Testing: Run all (especially visualization) examples and Open3D viewer with
        (Suzanne, Khronos helmet, large point cloud - apartment), on Linux x86-64,
        macOS (arm64), Windows x86-64.
-   [ ] PyPI: Upload wheels
-   [ ] Github: Publish release with auto tag version
-   [ ] Github: Create Open3D-ML release (auto tag version, auto-generate release notes)
-   [ ] `git pull` docs to website (Google cloud server `open3d:instance1`)
-   [ ] open3d.org (wordpress) : Update downloads table and post release notes
-   [ ] Upload release video to YouTube
-   [ ] Announce on Twitter, Discord, etc.

## @AGENT TODO [only after explicit user instruction]:
- ubuntu workflow uploads python 3.12 cpu wheel. Skip this.
- two linux devel packages for cpu and two viewers - one from ubuntu and one from ubuntu wheel. wheel viewer deb is 37MB while ubuntu viewer deb is 90MB
- devel packages have both include/tbb and include/oneapi/tbb - due to api change in tbb. perhaps leave this alone?
-  open3d-lib-sycl-shared 	21.9 GB  - why is this so big?
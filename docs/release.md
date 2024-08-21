# Open3D release process

## Release checklist

Collect all release artifacts in the [Github draft release page](https://github.com/isl-org/Open3D/releases)

-   [ ] Create release notes draft (auto-generate?)
-   [ ] Create release videos
-   [ ] Open3D-ML `dev_to_master` PR and merge
-   [ ] Version bump PR, merge to main
-   [ ] Build all Python version wheels (Linux, macOS, Windows x86_64) (CI)
-   [ ] Build Open3D C++ libraries (Linux, macOS, Windows x86_64)  (CI)
-   [ ] Build docs (CI) and push to [Open3d_website repo](https://github.com/isl-org/Open3D_website)
-   [ ] Build arm64 wheels (with ML Ops):
    -   [ ] macOS 12. Python 3.8+ (desktop):

        Configure: `cmake -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
                    -DBUILD_TENSORFLOW_OPS=ON -DBUILD_PYTORCH_OPS=ON -DBUNDLE_OPEN3D_ML=ON  ..`

    -   [ ] Ubuntu 20.04 Python (all versions) (follow docs/arm.rst) (desktop)

        Build command: `cd docker; ./docker_build.sh openblas-arm64-py311; ...`

-   [ ] Build Open3D app
    -   [ ] Ubuntu 20.04, Windows 10, macOS 10.15 x86_64: (CI)
    -   [ ] macOS 12 arm64 (desktop)
    -   [ ] macOS (x86_64, arm64) sign (desktop):

        From build/bin directory: `../../cpp/apps/sign_open3d_app.sh Open3D.app ../../cpp/apps/Open3DViewer/Open3dViewer.entitlements <apple-id>
                                    <cert-name> <team-id> <app-password>`

    -   [ ] (TBD) Windows app sign
-   [ ] Testing: Run all (especially visualization) examples and Open3D viewer with
        (Suzanne, Khronos helmet, large point cloud - apartment), on Linux x86-64,
        macOS (x86-64, arm64), Windows x86-64.
-   [ ] PyPI: Upload wheels
-   [ ] Github: Publish release with auto tag version
-   [ ] Github: Create Open3D-ML release (auto tag version, auto-generate release notes)
-   [ ] `git pull` docs to website (Google cloud server `open3d:instance1`)
-   [ ] open3d.org (wordpress) : Update downloads table and post release notes
-   [ ] Upload release video to YouTube
-   [ ] Announce on Twitter, Discord, etc.

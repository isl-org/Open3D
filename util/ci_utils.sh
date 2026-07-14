#!/usr/bin/env bash
set -euo pipefail

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

DEVELOPER_BUILD="${DEVELOPER_BUILD:-ON}"
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD="ON"
fi
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)} # POSIX: MacOS + Linux
NPROC=$((NPROC + 2))                         # run nproc+2 jobs to speed up the build
if [ -z "${BUILD_CUDA_MODULE:+x}" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        BUILD_CUDA_MODULE=ON
    else
        BUILD_CUDA_MODULE=OFF
    fi
fi
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-ON}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
LOW_MEM_USAGE=${LOW_MEM_USAGE:-OFF}
BUILD_SYCL_MODULE=${BUILD_SYCL_MODULE:-OFF}
BUILD_PYTHON_MODULE=${BUILD_PYTHON_MODULE:-ON}

# Dependency versions:
# CUDA: see docker/docker_build.sh
# ML
TENSORFLOW_VER="2.20.0"
TORCH_VER="2.10"
TORCH_REPO_URL="https://download.pytorch.org/whl/torch/"
# Python
PIP_VER="25.3"
PROTOBUF_VER="6.31.1"

OPEN3D_INSTALL_DIR=~/open3d_install
OPEN3D_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

install_python_dependencies() {

    echo "Installing Python dependencies"
    options="$(echo "$@" | tr ' ' '|')"
    python -m pip install -U pip=="$PIP_VER"
    python -m pip install -U -r "${OPEN3D_SOURCE_ROOT}/python/requirements_build.txt"
    if [[ "with-unit-test" =~ ^($options)$ ]]; then
        python -m pip install -U -r "${OPEN3D_SOURCE_ROOT}/python/requirements_test.txt"
    fi
    if [[ "with-cuda" =~ ^($options)$ ]]; then
        TF_ARCH_NAME=tensorflow
        TF_ARCH_DISABLE_NAME=tensorflow-cpu
        CUDA_VER=$(nvcc --version | grep "release " | cut -c33-37 | sed 's|[^0-9]||g') # e.g.: 117, 118, 121, ...
        TORCH_GLNX="torch==${TORCH_VER}+cu${CUDA_VER}"
    else
        # tensorflow-cpu wheels for macOS arm64 are not available
        if [[ "$OSTYPE" == "darwin"* ]]; then
            TF_ARCH_NAME=tensorflow
            TF_ARCH_DISABLE_NAME=tensorflow
        else
            TF_ARCH_NAME=tensorflow-cpu
            TF_ARCH_DISABLE_NAME=tensorflow
        fi
        TORCH_GLNX="torch==${TORCH_VER}+cpu"
    fi

    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt"
    if [[ "with-jupyter" =~ ^($options)$ ]]; then
        python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter_build.txt"
    fi

    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        # TF happily installs both CPU and GPU versions at the same time, so remove the other
        python -m pip uninstall --yes "$TF_ARCH_DISABLE_NAME"
        python -m pip install -U "$TF_ARCH_NAME"=="$TENSORFLOW_VER" # ML/requirements-tensorflow.txt
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then # ML/requirements-torch.txt
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            python -m pip install -U "${TORCH_GLNX}" -f "$TORCH_REPO_URL"
            python -m pip install tensorboard
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_VER" -f "$TORCH_REPO_URL" tensorboard
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -U -c "${OPEN3D_SOURCE_ROOT}/python/requirements_build.txt" yapf
        # Fix Protobuf compatibility issue
        # https://stackoverflow.com/a/72493690/1255535
        # https://github.com/protocolbuffers/protobuf/issues/10051
        python -m pip install -U protobuf=="$PROTOBUF_VER"
    fi
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        echo "Purge pip cache"
        python -m pip cache purge 2>/dev/null || true
    fi
}

build_all() {

    echo "Using cmake: $(command -v cmake)"
    cmake --version

    mkdir -p build
    cd build

    cmakeOptions=(
        -DDEVELOPER_BUILD="$DEVELOPER_BUILD"
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_CUDA_MODULE="$BUILD_CUDA_MODULE"
        -DBUILD_COMMON_CUDA_ARCHS=ON
        -DBUILD_COMMON_ISPC_ISAS=ON
        -DBUILD_TENSORFLOW_OPS="$BUILD_TENSORFLOW_OPS"
        -DBUILD_PYTORCH_OPS="$BUILD_PYTORCH_OPS"
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR"
        -DBUILD_UNIT_TESTS=ON
        -DBUILD_BENCHMARKS=ON
        -DBUILD_EXAMPLES=OFF
        -DBUILD_PYTHON_MODULE="$BUILD_PYTHON_MODULE"
    )

    echo
    echo Running cmake "${cmakeOptions[@]}" ..
    cmake "${cmakeOptions[@]}" ..
    echo
    echo "Build & install Open3D..."
    make VERBOSE=1 -j"$NPROC"
    make VERBOSE=1 install -j"$NPROC"
    if [[ "$BUILD_SHARED_LIBS" == "ON" ]]; then
        make package
    fi
    if [[ "$BUILD_PYTHON_MODULE" == "ON" ]]; then
        make VERBOSE=1 install-pip-package -j"$NPROC"
    fi
    echo
}

build_pip_package() {
    echo "Building Open3D wheel"
    options="$(echo "$@" | tr ' ' '|')"

    AARCH="$(uname -m)"
    if [[ "$AARCH" == "aarch64" ]]; then
        echo "Building for aarch64 architecture"
        BUILD_FILAMENT_FROM_SOURCE=ON
    else
        echo "Building for x86_64 architecture"
        BUILD_FILAMENT_FROM_SOURCE=OFF
    fi
    set +u
    if [[ -f "${OPEN3D_ML_ROOT}/set_open3d_ml_root.sh" ]] &&
        [[ "$BUILD_TENSORFLOW_OPS" == "ON" || "$BUILD_PYTORCH_OPS" == "ON" ]]; then
        echo "Open3D-ML available at ${OPEN3D_ML_ROOT}. Bundling Open3D-ML in wheel."
        # the build system of the main repo expects a main branch. make sure main exists
        git -C "${OPEN3D_ML_ROOT}" checkout -b main || true
        BUNDLE_OPEN3D_ML=ON
    else
        echo "Open3D-ML not available."
        BUNDLE_OPEN3D_ML=OFF
    fi
    if [[ "$DEVELOPER_BUILD" == "OFF" ]]; then
        echo "Building for a new Open3D release"
    fi
    if [[ "build_azure_kinect" =~ ^($options)$ ]]; then
        echo "Azure Kinect enabled in Python wheel."
        BUILD_AZURE_KINECT=ON
    else
        echo "Azure Kinect disabled in Python wheel."
        BUILD_AZURE_KINECT=OFF
    fi
    if [[ "build_jupyter" =~ ^($options)$ ]]; then
        echo "Building Jupyter extension in Python wheel."
        BUILD_JUPYTER_EXTENSION=ON
        BUILD_WEBRTC_FROM_SOURCE=ON
    else
        echo "Jupyter extension disabled in Python wheel."
        BUILD_JUPYTER_EXTENSION=OFF
        BUILD_WEBRTC_FROM_SOURCE=OFF
    fi
    set -u

    echo
    mkdir -p build
    pushd build
    cmakeOptions=("-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DBUILD_COMMON_ISPC_ISAS=ON"
        "-DBUILD_AZURE_KINECT=$BUILD_AZURE_KINECT"
        "-DBUILD_LIBREALSENSE=ON"
        "-DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS"
        "-DBUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS"
        "-DBUILD_FILAMENT_FROM_SOURCE=$BUILD_FILAMENT_FROM_SOURCE"
        "-DBUILD_JUPYTER_EXTENSION=$BUILD_JUPYTER_EXTENSION"
        "-DBUILD_WEBRTC=$BUILD_WEBRTC_FROM_SOURCE"
        "-DCMAKE_INSTALL_PREFIX=$OPEN3D_INSTALL_DIR"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_UNIT_TESTS=OFF"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUNDLE_OPEN3D_ML=$BUNDLE_OPEN3D_ML"
        "-DBUILD_SHARED_LIBS=ON"
    )
    if [ "$BUILD_CUDA_MODULE" == ON ]; then
        install_python_dependencies with-cuda purge-cache
        cmakeOptions+=("-DBUILD_CUDA_MODULE=ON" "-DBUILD_COMMON_CUDA_ARCHS=ON")
    else
        cmakeOptions+=("-DBUILD_CUDA_MODULE=OFF")
    fi
    set -x
    # Always rerun cmake and unset Python cache variables to update cache for
    # the current Python and build options.  This avoids incorrect/inherited
    # Python config and ensures all paths are correct.  Do not use --fresh: it
    # unnecessarily wipes build objects and wastes disk/CI time.
    cmake -U 'Python3*' -U 'PYTHON_*' -U 'Pytorch*' -U 'Torch*' \
        -DBUILD_PYTHON_MODULE="${BUILD_PYTHON_MODULE}" \
        -DPython3_EXECUTABLE="$(command -v python3)" \
        "${cmakeOptions[@]}" ..
    set +x
    if [ "$BUILD_PYTHON_MODULE" == "OFF" ]; then
        echo "Building Open3D C++ Core only..."
        make VERBOSE=1 -j"$NPROC"
    else
        echo "Packaging Open3D pip wheel (single configure)..."
        make VERBOSE=1 -j"$NPROC" pip-package
    fi
    popd

    # A CUDA-enabled build also gets a companion CPU-only "open3d-cpu" wheel
    # built alongside it (for users without a CUDA GPU); a CPU-only build IS
    # already the CPU wheel, so there is nothing extra to build in that case.
    # Prefer build_pip_package_from_installed() in CI (separate devel prefixes).
    if [ "$BUILD_PYTHON_MODULE" != "OFF" ] && [ "$BUILD_CUDA_MODULE" == ON ]; then
        echo
        echo "Building open3d-cpu wheel (reusing single build folder)..."
        # 1. Back up the CUDA-enabled wheel(s) to a temporary directory outside build/
        mkdir -p pip_package_backup
        cp build/lib/python_package/pip_package/*.whl pip_package_backup/

        # 2. Reconfigure the existing build directory with CUDA disabled
        pushd build
        # We must make sure BUILD_CUDA_MODULE=OFF overrides any ON from cmakeOptions.
        # Repeating -D left-to-right makes the last one win, so we place it after cmakeOptions.
        set -x
        cmake "${cmakeOptions[@]}" -DBUILD_CUDA_MODULE=OFF ..
        set +x

        # 3. Build the CPU companion wheel
        make VERBOSE=1 -j"$NPROC" pip-package
        popd

        # 4. Restore the backed-up CUDA wheel(s) into the pip_package directory
        mv pip_package_backup/*.whl build/lib/python_package/pip_package/
        rm -rf pip_package_backup
    fi
    echo
}

# Selective wipe of Open3D outputs so a CPU→CUDA (or reverse) reconfigure can
# reuse 3rdparty ExternalProjects. Keeps filament/assimp/embree/vtk build trees.
_open3d_wipe_compiled_outputs() {
    local build_dir="${1:-build}"
    echo "Removing Open3D compiled outputs under ${build_dir} (keeping 3rdparty)"
    rm -rf \
        "${build_dir}/bin" \
        "${build_dir}/cpp" \
        "${build_dir}/lib/_build_config.py" \
        "${build_dir}/lib/ml" \
        "${build_dir}/lib/python_package" \
        "${build_dir}/lib/cmake" \
        "${build_dir}/package" \
        "${build_dir}/package-Open3DViewer-deb" \
        "${build_dir}/Open3D"
    # Shared lib + static archives produced by the Open3D targets (not 3rdparty)
    rm -f "${build_dir}"/lib/Release/libOpen3D.so* \
        "${build_dir}"/lib/Release/*.a \
        "${build_dir}"/lib/libOpen3D.so* 2>/dev/null || true
}

# Build CPU then CUDA open3d-devel packages (+ viewer deb + C++ tests bundle).
# ML ops are OFF (Python/Torch/TF ABI belongs in the wheel stage).
# Artifacts are copied to OPEN3D_ARTIFACTS_DIR (default /open3d-artifacts).
# Does not run ./bin/tests — use export_cpp_tests_bundle + run_cpp_tests_from_bundle.
build_devel_packages_cpu_cuda() {
    local artifacts_dir="${OPEN3D_ARTIFACTS_DIR:-/open3d-artifacts}"
    mkdir -p "${artifacts_dir}"

    echo "Building Open3D devel packages (CPU then CUDA) → ${artifacts_dir}"
    echo "Using cmake: $(command -v cmake)"
    cmake --version

    local cmakeOptions=(
        "-DDEVELOPER_BUILD=${DEVELOPER_BUILD}"
        "-DBUILD_SHARED_LIBS=ON"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_PYTHON_MODULE=OFF"
        "-DBUILD_TENSORFLOW_OPS=OFF"
        "-DBUILD_PYTORCH_OPS=OFF"
        "-DBUNDLE_OPEN3D_ML=OFF"
        "-DBUILD_UNIT_TESTS=ON"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUILD_EXAMPLES=OFF"
        "-DBUILD_GUI=ON"
        "-DBUILD_LIBREALSENSE=ON"
        "-DBUILD_AZURE_KINECT=ON"
        "-DBUILD_COMMON_ISPC_ISAS=ON"
        "-DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR}"
    )

    mkdir -p build
    pushd build

    echo
    echo "=== CPU devel package ==="
    set -x
    cmake -DBUILD_CUDA_MODULE=OFF "${cmakeOptions[@]}" ..
    set +x
    make VERBOSE=1 -j"$NPROC"
    make VERBOSE=1 Open3DViewer -j"$NPROC"
    # CPack stages its own install tree; avoid a separate make install that can
    # fail on unrelated FetchContent targets (e.g. gmock) while still packaging
    # Open3D's install() rules.
    make package
    make package-Open3DViewer-deb

    # Stage CPU artifacts before the CUDA flip overwrites package/
    shopt -s nullglob
    cp -a package/open3d-devel-*.tar.xz "${artifacts_dir}/"
    cp -a package-Open3DViewer-deb/open3d-viewer-*-Linux.deb "${artifacts_dir}/" 2>/dev/null \
        || cp -a package-Open3DViewer-deb/*.deb "${artifacts_dir}/"
    shopt -u nullglob

    export_cpp_tests_bundle "${artifacts_dir}"
    popd

    echo
    echo "=== CUDA devel package (reusing 3rdparty) ==="
    _open3d_wipe_compiled_outputs build
    pushd build
    set -x
    cmake -DBUILD_CUDA_MODULE=ON -DBUILD_COMMON_CUDA_ARCHS=ON "${cmakeOptions[@]}" ..
    set +x
    make VERBOSE=1 -j"$NPROC"
    make package
    shopt -s nullglob
    cp -a package/open3d-devel-*-cuda-*.tar.xz "${artifacts_dir}/"
    shopt -u nullglob
    popd

    echo "Devel artifacts:"
    ls -lh "${artifacts_dir}"
}

# Package bin/tests + runtime libs needed to run C++ unit tests out-of-tree.
# Usage: export_cpp_tests_bundle [artifacts_dir]  (must run from build/ or pass paths)
export_cpp_tests_bundle() {
    local artifacts_dir="${1:-${OPEN3D_ARTIFACTS_DIR:-/open3d-artifacts}}"
    local build_dir="${OPEN3D_BUILD_DIR:-}"
    if [[ -z "${build_dir}" ]]; then
        if [[ -f bin/tests ]]; then
            build_dir="."
        elif [[ -f build/bin/tests ]]; then
            build_dir="build"
        else
            echo "export_cpp_tests_bundle: bin/tests not found"
            exit 1
        fi
    fi
    mkdir -p "${artifacts_dir}"
    local staging
    staging="$(mktemp -d)"
    mkdir -p "${staging}/bin" "${staging}/lib"
    cp -a "${build_dir}/bin/tests" "${staging}/bin/"
    # Runtime deps for shared libOpen3D
    shopt -s nullglob
    cp -a "${build_dir}/lib/Release/libOpen3D.so"* "${staging}/lib/" 2>/dev/null || true
    # Bundled TBB next to tests (rpath / LD_LIBRARY_PATH)
    if [[ -d "${build_dir}/gnu_"* ]]; then
        # shellcheck disable=SC2086
        cp -a ${build_dir}/gnu_*/libtbb.so* "${staging}/lib/" 2>/dev/null || true
    fi
    find "${build_dir}" -maxdepth 2 -name 'libtbb.so*' -exec cp -a {} "${staging}/lib/" \; 2>/dev/null || true
    # GUI resources referenced by some tests
    if [[ -d "${build_dir}/bin/resources" ]]; then
        cp -a "${build_dir}/bin/resources" "${staging}/bin/"
    fi
    shopt -u nullglob
    local out="${artifacts_dir}/open3d-cpp-tests.tar.xz"
    tar -C "${staging}" -cJf "${out}" bin lib
    rm -rf "${staging}"
    echo "Wrote ${out}"
}

# Extract a tests bundle and run ./bin/tests.
# Usage: run_cpp_tests_from_bundle /path/to/open3d-cpp-tests.tar.xz [gtest args...]
run_cpp_tests_from_bundle() {
    local bundle="$1"
    shift || true
    local work
    work="$(mktemp -d)"
    tar -C "${work}" -xf "${bundle}"
    export LD_LIBRARY_PATH="${work}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    pushd "${work}/bin"
    echo "Run ./tests $* --gtest_random_seed=SEED to repeat this test sequence."
    ./tests "$@"
    popd
    rm -rf "${work}"
}

# Build open3d (+ open3d-cpu companion) wheels against extracted devel prefixes.
# Requires:
#   OPEN3D_CUDA_ROOT  - extracted CUDA open3d-devel prefix
#   OPEN3D_CPU_ROOT   - extracted CPU open3d-devel prefix
# Optional args: build_azure_kinect build_jupyter (same as build_pip_package)
build_pip_package_from_installed() {
    echo "Building Open3D wheels from installed devel packages"
    options="$(echo "$@" | tr ' ' '|')"

    if [[ -z "${OPEN3D_CUDA_ROOT:-}" || -z "${OPEN3D_CPU_ROOT:-}" ]]; then
        echo "OPEN3D_CUDA_ROOT and OPEN3D_CPU_ROOT must point at extracted devel prefixes"
        exit 1
    fi
    if [[ ! -d "${OPEN3D_CUDA_ROOT}" || ! -d "${OPEN3D_CPU_ROOT}" ]]; then
        echo "Missing devel prefix(es): CUDA=${OPEN3D_CUDA_ROOT} CPU=${OPEN3D_CPU_ROOT}"
        exit 1
    fi

    AARCH="$(uname -m)"
    if [[ "$AARCH" == "aarch64" ]]; then
        BUILD_FILAMENT_FROM_SOURCE=ON
    else
        BUILD_FILAMENT_FROM_SOURCE=OFF
    fi
    set +u
    if [[ -f "${OPEN3D_ML_ROOT}/set_open3d_ml_root.sh" ]] &&
        [[ "$BUILD_TENSORFLOW_OPS" == "ON" || "$BUILD_PYTORCH_OPS" == "ON" ]]; then
        echo "Open3D-ML available at ${OPEN3D_ML_ROOT}. Bundling Open3D-ML in wheel."
        git -C "${OPEN3D_ML_ROOT}" checkout -b main || true
        BUNDLE_OPEN3D_ML=ON
    else
        echo "Open3D-ML not available."
        BUNDLE_OPEN3D_ML=OFF
    fi
    if [[ "build_azure_kinect" =~ ^($options)$ ]]; then
        BUILD_AZURE_KINECT=ON
    else
        BUILD_AZURE_KINECT=OFF
    fi
    if [[ "build_jupyter" =~ ^($options)$ ]]; then
        BUILD_JUPYTER_EXTENSION=ON
        BUILD_WEBRTC_FLAG=ON
    else
        BUILD_JUPYTER_EXTENSION=OFF
        BUILD_WEBRTC_FLAG=OFF
    fi
    set -u

    # Match main's build_pip_package() ML-ops ABI while building the CPU and CUDA
    # wheels in separate dirs. The wheel image already ships CPU torch/tf, so the
    # CPU wheel links its ops against CPU torch. Before the CUDA wheel we install
    # CUDA torch/tf (see below) so its ops link against CUDA torch. Both wheels
    # build torch/tf ops per BUILD_{PYTORCH,TENSORFLOW}_OPS.
    if [[ "$BUILD_PYTORCH_OPS" == "ON" || "$BUILD_TENSORFLOW_OPS" == "ON" ]] &&
        ! python -c "import torch" >/dev/null 2>&1; then
        install_python_dependencies purge-cache
    fi

    local commonOptions=(
        "-DOPEN3D_USE_INSTALLED_LIBRARY=ON"
        "-DDEVELOPER_BUILD=${DEVELOPER_BUILD}"
        "-DBUILD_SHARED_LIBS=ON"
        "-DBUILD_PYTHON_MODULE=ON"
        "-DBUILD_AZURE_KINECT=${BUILD_AZURE_KINECT}"
        "-DBUILD_LIBREALSENSE=ON"
        "-DBUILD_JUPYTER_EXTENSION=${BUILD_JUPYTER_EXTENSION}"
        "-DBUILD_WEBRTC=${BUILD_WEBRTC_FLAG}"
        "-DBUILD_GUI=ON"
        "-DBUILD_UNIT_TESTS=OFF"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUILD_EXAMPLES=OFF"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DPython3_EXECUTABLE=$(command -v python3)"
    )

    local wheel_out="build_wheel_out"
    mkdir -p "${wheel_out}"

    # CPU companion wheel first (same torch ABI as the CUDA wheel below).
    echo
    echo "=== CPU companion wheel (Open3D_ROOT=${OPEN3D_CPU_ROOT}) ==="
    mkdir -p build_cpu_wheel
    pushd build_cpu_wheel
    set -x
    cmake -U 'Python3*' -U 'PYTHON_*' -U 'Pytorch*' -U 'Torch*' \
        -DOpen3D_ROOT="${OPEN3D_CPU_ROOT}" \
        -DCMAKE_PREFIX_PATH="${OPEN3D_CPU_ROOT}" \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        -DBUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        -DBUNDLE_OPEN3D_ML="${BUNDLE_OPEN3D_ML}" \
        "${commonOptions[@]}" \
        ..
    set +x
    make VERBOSE=1 -j"$NPROC" pip-package
    cp -a lib/python_package/pip_package/open3d*.whl "../${wheel_out}/"
    popd

    # Install CUDA torch/tf so the CUDA wheel's ops link against CUDA torch
    # (matches main's build_pip_package). The CPU wheel above is already built
    # and saved, so replacing torch in the env now is safe.
    if [[ "$BUILD_PYTORCH_OPS" == "ON" || "$BUILD_TENSORFLOW_OPS" == "ON" ]]; then
        echo "Installing CUDA versions of TensorFlow and PyTorch..."
        install_python_dependencies with-cuda purge-cache
    fi

    echo
    echo "=== CUDA wheel (Open3D_ROOT=${OPEN3D_CUDA_ROOT}) ==="
    mkdir -p build_cuda_wheel
    pushd build_cuda_wheel
    set -x
    # Unset cached Python/Torch entries: torch changed to the CUDA build above.
    cmake -U 'Python3*' -U 'PYTHON_*' -U 'Pytorch*' -U 'Torch*' \
        -DOpen3D_ROOT="${OPEN3D_CUDA_ROOT}" \
        -DCMAKE_PREFIX_PATH="${OPEN3D_CUDA_ROOT}" \
        -DBUILD_CUDA_MODULE=ON \
        -DBUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        -DBUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        -DBUNDLE_OPEN3D_ML="${BUNDLE_OPEN3D_ML}" \
        "${commonOptions[@]}" \
        ..
    set +x
    # Guard: installed mode must not rebuild heavy 3rdparty ExternalProjects
    if [[ -d assimp || -d embree || -d filament || -d vtk ]]; then
        echo "ERROR: 3rdparty ExternalProject dirs present in installed-mode build"
        exit 1
    fi
    make VERBOSE=1 -j"$NPROC" pip-package
    cp -a lib/python_package/pip_package/open3d*.whl "../${wheel_out}/"
    popd

    # Place wheels where docker_build.sh / CI expect them
    mkdir -p build/lib/python_package/pip_package
    cp -a "${wheel_out}"/*.whl build/lib/python_package/pip_package/
    echo "Wheels built:"
    ls -lh build/lib/python_package/pip_package/
}

# Extract open3d-devel tar.xz into a prefix directory.
# Usage: extract_open3d_devel /path/to/open3d-devel-....tar.xz /opt/open3d-cuda
extract_open3d_devel() {
    local tar_path="$1"
    local dest="$2"
    mkdir -p "${dest}"
    # CPack TXZ typically has a single top-level directory; strip it.
    tar -C "${dest}" --strip-components=1 -xf "${tar_path}"
    echo "Extracted ${tar_path} → ${dest}"
    ls "${dest}/lib/cmake/Open3D" >/dev/null
}

# Test wheel in blank virtual environment
# Usage: test_wheel wheel_path
test_wheel() {
    wheel_path="$1"
    python -m venv open3d_test.venv
    # shellcheck disable=SC1091
    source open3d_test.venv/bin/activate
    python -m pip install -U pip=="$PIP_VER"
    python -m pip install -U -c "${OPEN3D_SOURCE_ROOT}/python/requirements_build.txt" wheel setuptools
    echo -n "Using python: $(command -v python)"
    python --version
    echo -n "Using pip: "
    python -m pip --version
    echo "Installing Open3D wheel $wheel_path in virtual environment..."
    python -m pip install "$wheel_path"
    python -W default -c "import open3d; print('Installed:', open3d); print('BUILD_CUDA_MODULE: ', open3d._build_config['BUILD_CUDA_MODULE'])"
    python -W default -c "import open3d; print('CUDA available: ', open3d.core.cuda.is_available())"
    echo
    # echo "Dynamic libraries used:"
    # DLL_PATH=$(dirname $(python -c "import open3d; print(open3d.cpu.pybind.__file__)"))/..
    # if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    #     find "$DLL_PATH"/{cpu,cuda}/ -type f -print -execdir ldd {} \;
    # elif [[ "$OSTYPE" == "darwin"* ]]; then
    #     find "$DLL_PATH"/cpu/ -type f -execdir otool -L {} \;
    # fi
    echo
    HAVE_PYTORCH_OPS=OFF
    HAVE_TENSORFLOW_OPS=OFF
    if python -c "import sys, open3d; sys.exit(not open3d._build_config['BUILD_PYTORCH_OPS'])"; then
        HAVE_PYTORCH_OPS=ON
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-torch.txt"
        python -W default -c \
            "import open3d.ml.torch; print('PyTorch Ops library loaded:', open3d.ml.torch._loaded)"
    fi
    if python -c "import sys, open3d; sys.exit(not open3d._build_config['BUILD_TENSORFLOW_OPS'])"; then
        HAVE_TENSORFLOW_OPS=ON
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-tensorflow.txt"
        python -W default -c \
            "import open3d.ml.tf.ops; print('TensorFlow Ops library loaded:', open3d.ml.tf.ops)"
    fi
    if [ "$HAVE_TENSORFLOW_OPS" == ON ] && [ "$HAVE_PYTORCH_OPS" == ON ]; then
        echo "Importing TensorFlow and torch in the reversed order"
        python -W default -c "import tensorflow as tf; import torch; import open3d.ml.torch as o3d"
        echo "Importing TensorFlow and torch in the normal order"
        python -W default -c "import open3d.ml.torch as o3d; import tensorflow as tf; import torch"
    fi
    deactivate open3d_test.venv # argument prevents unbound variable error
}

# Run in virtual environment
run_python_tests() {
    # shellcheck disable=SC1091
    source open3d_test.venv/bin/activate
    python -m pip install -U -r "$OPEN3D_SOURCE_ROOT/python/requirements_test.txt"
    echo Add --randomly-seed=SEED to the test command to reproduce test order.
    pytest_args=("$OPEN3D_SOURCE_ROOT"/python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] && [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore "$OPEN3D_SOURCE_ROOT"/python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
    deactivate open3d_test.venv # argument prevents unbound variable error
    rm -rf open3d_test.venv     # cleanup for testing the next wheel
}

# Use: run_unit_tests
run_cpp_unit_tests() {
    unitTestFlags=--gtest_shuffle
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    echo "Run ./bin/tests $unitTestFlags --gtest_random_seed=SEED to repeat this test sequence."
    ./bin/tests "$unitTestFlags"
    echo
}

# test_cpp_example runExample
# Need variable OPEN3D_INSTALL_DIR
test_cpp_example() {
    # Now I am in Open3D/build/
    pushd ../examples/cmake/open3d-cmake-find-package
    mkdir build
    pushd build
    echo Testing build with cmake
    cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
    make -j"$NPROC" VERBOSE=1
    runExample="$1"
    if [ "$runExample" == ON ]; then
        ./Draw --skip-for-unit-test
    fi
    if [ "$BUILD_SHARED_LIBS" == ON ]; then
        rm -r ./*
        echo Testing build with pkg-config
        export PKG_CONFIG_PATH=${OPEN3D_INSTALL_DIR}/lib/pkgconfig
        echo Open3D build options: $(pkg-config --cflags --libs Open3D)
        c++ ../Draw.cpp -o Draw $(pkg-config --cflags --libs Open3D)
        if [ "$runExample" == ON ]; then
            ./Draw --skip-for-unit-test
        fi
    fi
    popd
    popd
    # Now I am in Open3D/build/
}

# Install dependencies needed for building documentation. This no longer
# needs a compiler toolchain: docs are built against a pre-built Open3D wheel
# (see build_docs() below), which already includes GUI/rendering and, on
# Linux, GPU-accelerated EGL offscreen rendering for notebook execution.
# Usage: install_docs_dependencies "${OPEN3D_ML_ROOT}"
install_docs_dependencies() {
    echo
    echo Install ubuntu dependencies from $(pwd)
    # Provides runtime OpenGL/EGL libraries needed by the legacy visualizer
    # for GPU-accelerated headless rendering of notebook outputs.
    util/install_deps_ubuntu.sh assume-yes
    $SUDO apt-get install --yes \
        libxml2-dev libxslt-dev \
        python3-dev python-is-python3 python3-pip \
        doxygen \
        texlive \
        texlive-latex-extra \
        ghostscript \
        pandoc
    echo
    echo Install Python dependencies for building docs
    command -v python
    python -V
    python -m pip install -U -q "pip==$PIP_VER"
    if [[ -d "$1" ]]; then
        OPEN3D_ML_ROOT="$1"
        echo Installing Open3D-ML dependencies from "${OPEN3D_ML_ROOT}"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements.txt" \
            -r "${OPEN3D_ML_ROOT}/requirements-tensorflow.txt" \
            -r "${OPEN3D_ML_ROOT}/requirements-torch.txt"
    else
        echo OPEN3D_ML_ROOT="$OPEN3D_ML_ROOT" not specified or invalid. Skipping ML dependencies.
    fi
    echo
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt" \
        -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter_build.txt" \
        -r "${OPEN3D_SOURCE_ROOT}/docs/requirements.txt"
}

# Build documentation (Sphinx + Doxygen + Jupyter notebooks) against an
# already-installed Open3D Python wheel (built by the standard cpu-shared-ml
# job; see .github/workflows/ubuntu.yml). A single pass now covers notebook
# execution and visualization.{gui,rendering} API docs, since the standard
# binary supports both GUI and (on Linux) GPU-accelerated EGL headless
# rendering; there is no separate headless build.
# Usage: build_docs $DEVELOPER_BUILD
build_docs() {
    set +u
    DEVELOPER_BUILD="$1"
    set -u
    if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
        DOC_ARGS=""
    else
        DOC_ARGS="--is_release"
        echo "Building docs for a new Open3D release"
    fi
    python -c "from open3d import *; import open3d; print(open3d)"
    set -x # Echo commands on
    cd "${OPEN3D_SOURCE_ROOT}/docs" # Works regardless of caller's cwd.
    # docs/Doxyfile, getting_started.rst and docker.rst are normally generated
    # by docs/CMakeLists.txt's configure_file() calls during a full `cmake ..`
    # configure. Since this build only installs a pre-built wheel (no cmake
    # configure step), generate them here instead by substituting
    # @OPEN3D_VERSION_FULL@ (full version incl. dev hash) and @OPEN3D_VERSION@
    # (release version only), taken from the installed wheel.
    OPEN3D_VERSION_FULL="$(python -c 'import open3d; print(open3d.__version__)')"
    OPEN3D_VERSION="${OPEN3D_VERSION_FULL%%+*}"
    subst_version() { # subst_version <input> <output>
        sed -e "s|@OPEN3D_VERSION_FULL@|${OPEN3D_VERSION_FULL}|g" \
            -e "s|@OPEN3D_VERSION@|${OPEN3D_VERSION}|g" "$1" >"$2"
    }
    subst_version Doxyfile.in Doxyfile
    subst_version getting_started.in.rst getting_started.rst
    subst_version docker.in.rst docker.rst
    python make_docs.py $DOC_ARGS --clean_notebooks --execute_notebooks=always \
        --py_api_rst=always --py_example_rst=always --sphinx --doxygen
    cd - >/dev/null
    set +x # Echo commands off
}

maximize_ubuntu_github_actions_build_space() {
    # https://github.com/easimon/maximize-build-space/blob/main/action.yml
    df -h .                                  # => 26GB
    $SUDO rm -rf /usr/share/dotnet           # ~17GB
    $SUDO rm -rf /usr/local/lib/android      # ~11GB
    $SUDO rm -rf /opt/ghc                    # ~2.7GB
    $SUDO rm -rf /opt/hostedtoolcache/CodeQL # ~5.4GB
    $SUDO docker image prune --all --force   # ~4.5GB
    $SUDO rm -rf "$AGENT_TOOLSDIRECTORY"
    df -h . # => 53GB
}

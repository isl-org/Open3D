# Exports: ${STDGPU_INCLUDE_DIRS}
# Exports: ${STDGPU_LIB_DIR}
# Exports: ${STDGPU_LIBRARIES}

include(ExternalProject)

find_package(Git QUIET REQUIRED)

if(USE_HIP)
    # stdgpu's pinned commit ships an experimental HIP backend
    # (STDGPU_BACKEND_HIP). Build it with hipcc/clang: the device headers are
    # marked LANGUAGE HIP by stdgpu itself, and the host impl/*.cpp reach the
    # HIP backend headers (rocPRIM, clang-only builtins) so the CXX compiler
    # must also be clang. stdgpu's bundled cmake/hip/Findthrust.cmake locates
    # rocThrust and exposes thrust::thrust, so no Findthrust shim is needed (the
    # broken bundled Findthrust in the older stdgpu 1.3.0 release is fixed at
    # this pinned commit).
    ExternalProject_Add(
        ext_stdgpu
        PREFIX stdgpu
        URL https://github.com/stotko/stdgpu/archive/d7c07d056a654454c3f2d7cf928e6cec6e0ce28e.tar.gz
        URL_HASH SHA256=85b38dc3807ac24b5afe8d06a674dbff8e66579dfd67cbb33189783b0a84a0aa
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/stdgpu"
        UPDATE_COMMAND ""
        # The pinned commit's HIP backend dir is missing determine_thrust_paths
        # .cmake (only the CUDA backend ships it), but src/stdgpu/CMakeLists.txt
        # unconditionally installs it -> the install step fails. Open3D consumes
        # stdgpu by include-dir + lib (not find_package(stdgpu)), so the file is
        # never used; copy the CUDA one into the hip dir so install succeeds.
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different
            <SOURCE_DIR>/cmake/cuda/determine_thrust_paths.cmake
            <SOURCE_DIR>/cmake/hip/determine_thrust_paths.cmake
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP
            -DSTDGPU_BUILD_SHARED_LIBS=OFF
            -DSTDGPU_BUILD_EXAMPLES=OFF
            -DSTDGPU_BUILD_TESTS=OFF
            -DSTDGPU_BUILD_BENCHMARKS=OFF
            -DSTDGPU_ENABLE_CONTRACT_CHECKS=OFF
            -DSTDGPU_COMPILE_WARNING_AS_ERROR=OFF
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            ${ExternalProject_CMAKE_ARGS_hidden}
            # These must come AFTER the hidden args, which set CMAKE_CXX_COMPILER
            # to the host g++: stdgpu's HIP backend reaches rocPRIM (clang-only
            # builtins) from its host impl/*.cpp, so CXX must be hipcc/clang too.
            -DCMAKE_CXX_COMPILER=${CMAKE_HIP_COMPILER}
            -DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
        CMAKE_CACHE_ARGS
            -DCMAKE_HIP_ARCHITECTURES:STRING=${CMAKE_HIP_ARCHITECTURES}
        BUILD_BYPRODUCTS
            <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}stdgpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Get_Property(ext_stdgpu INSTALL_DIR)
    set(STDGPU_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(STDGPU_LIB_DIR ${INSTALL_DIR}/lib)
    set(STDGPU_LIBRARIES stdgpu)
    return()
endif()

# In CUDA 13.0+, Thrust moved from include/thrust/ to include/cccl/thrust/
# Set the appropriate include directory based on CUDA version
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(THRUST_INCLUDE_DIR_PATH "${CUDAToolkit_LIBRARY_ROOT}/include/cccl")
    # In CUDA 13.0+, Thrust renamed is_proxy_reference to is_wrapped_reference
    # Initialize git repository and apply patch for Thrust API compatibility
    set(STDGPU_PATCH_COMMAND
        COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} init
        COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} add -A
        COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_LIST_DIR}/fix-thrust-is-proxy-reference.patch
    )
else()
    set(THRUST_INCLUDE_DIR_PATH "${CUDAToolkit_LIBRARY_ROOT}/include")
    set(STDGPU_PATCH_COMMAND "")
endif()

ExternalProject_Add(
    ext_stdgpu
    PREFIX stdgpu
    URL https://github.com/stotko/stdgpu/archive/d7c07d056a654454c3f2d7cf928e6cec6e0ce28e.tar.gz
    URL_HASH SHA256=85b38dc3807ac24b5afe8d06a674dbff8e66579dfd67cbb33189783b0a84a0aa
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/stdgpu"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${STDGPU_PATCH_COMMAND}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DSTDGPU_BUILD_SHARED_LIBS=OFF
        -DSTDGPU_BUILD_EXAMPLES=OFF
        -DSTDGPU_BUILD_TESTS=OFF
        -DSTDGPU_BUILD_BENCHMARKS=OFF
        -DSTDGPU_ENABLE_CONTRACT_CHECKS=OFF
        -DTHRUST_INCLUDE_DIR=${THRUST_INCLUDE_DIR_PATH}
        ${ExternalProject_CMAKE_ARGS_hidden}
    CMAKE_CACHE_ARGS    # Lists must be passed via CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}stdgpu${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_stdgpu INSTALL_DIR)
set(STDGPU_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(STDGPU_LIB_DIR ${INSTALL_DIR}/lib)
set(STDGPU_LIBRARIES stdgpu)

# Exports: ${STDGPU_INCLUDE_DIRS}
# Exports: ${STDGPU_LIB_DIR}
# Exports: ${STDGPU_LIBRARIES}

include(ExternalProject)

# In CUDA 13.0+, Thrust moved from include/thrust/ to include/cccl/thrust/
# Set the appropriate include directory based on CUDA version
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(THRUST_INCLUDE_DIR_PATH "${CUDAToolkit_LIBRARY_ROOT}/include/cccl")
else()
    set(THRUST_INCLUDE_DIR_PATH "${CUDAToolkit_LIBRARY_ROOT}/include")
endif()

set(STDGPU_CMAKE_ARGS "")
if(MSVC AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2")
    # CUDA 13.2 CCCL headers require MSVC's standard-conforming preprocessor.
    list(APPEND STDGPU_CMAKE_ARGS
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} /Zc:preprocessor"
        "-DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG} /Zc:preprocessor"
        "-DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE} /Zc:preprocessor"
        "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zc:preprocessor"
        "-DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL} /Zc:preprocessor"
    )
endif()

ExternalProject_Add(
    ext_stdgpu
    PREFIX stdgpu
    URL https://github.com/stotko/stdgpu/archive/0bebd1f51e686cb634461ab3a5271fb3c04560c4.tar.gz
    URL_HASH SHA256=7c3b80cdc1715adbcb6d5f5af29beb5953730cb8357fe15b79281ec439934434
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/stdgpu"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DSTDGPU_BUILD_SHARED_LIBS=OFF
        -DSTDGPU_BUILD_EXAMPLES=OFF
        -DSTDGPU_BUILD_TESTS=OFF
        -DSTDGPU_BUILD_BENCHMARKS=OFF
        -DSTDGPU_ENABLE_CONTRACT_CHECKS=OFF
        -DTHRUST_INCLUDE_DIR=${THRUST_INCLUDE_DIR_PATH}
        ${STDGPU_CMAKE_ARGS}
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

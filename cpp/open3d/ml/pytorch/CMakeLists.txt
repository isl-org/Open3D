message(STATUS "Building Pytorch ops")

if(BUILD_CUDA_MODULE)
    message(STATUS "Building Pytorch ops with CUDA")
endif()

find_package(Pytorch REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(BUILD_CUDA_MODULE)
    find_package(CUB REQUIRED)
    find_package(CUTLASS REQUIRED)
endif()

set(TORCH_OPS_SOURCES
    "continuous_conv/ContinuousConvBackpropFilterOpKernel.cpp"
    "continuous_conv/ContinuousConvOpKernel.cpp"
    "continuous_conv/ContinuousConvOps.cpp"
    "continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cpp"
    "continuous_conv/ContinuousConvTransposeOpKernel.cpp"
    "continuous_conv/ContinuousConvTransposeOps.cpp"
    "misc/BuildSpatialHashTableOpKernel.cpp"
    "misc/InvertNeighborsListOps.cpp"
    "misc/RadiusSearchOpKernel.cpp"
    "misc/ReduceSubarraysSumOps.cpp"
    "misc/FixedRadiusSearchOps.cpp"
    "misc/KnnSearchOpKernel.cpp"
    "misc/RadiusSearchOps.cpp"
    "misc/VoxelPoolingOpKernel.cpp"
    "misc/BuildSpatialHashTableOps.cpp"
    "misc/InvertNeighborsListOpKernel.cpp"
    "misc/KnnSearchOps.cpp"
    "misc/ReduceSubarraysSumOpKernel.cpp"
    "misc/VoxelPoolingOps.cpp"
    "misc/FixedRadiusSearchOpKernel.cpp"
)

set(TORCH_OPS_CUDA_SOURCES
    "continuous_conv/ContinuousConvBackpropFilterOpKernel.cu"
    "continuous_conv/ContinuousConvOpKernel.cu"
    "continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cu"
    "continuous_conv/ContinuousConvTransposeOpKernel.cu"
    "misc/FixedRadiusSearchOpKernel.cu"
    "misc/BuildSpatialHashTableOpKernel.cu"
    "misc/InvertNeighborsListOpKernel.cu"
    "misc/ReduceSubarraysSumOpKernel.cu"
    "../impl/continuous_conv/ContinuousConvCUDAKernels.cu"
)

if(BUILD_CUDA_MODULE)
    set(open3d_torch_ops_SOURCES
        ${TORCH_OPS_SOURCES}
        ${TORCH_OPS_CUDA_SOURCES}
    )
else()
    set(open3d_torch_ops_SOURCES
        ${TORCH_OPS_SOURCES}
    )
endif()

add_library(open3d_torch_ops SHARED ${open3d_torch_ops_SOURCES})
open3d_set_global_properties(open3d_torch_ops)
open3d_show_and_abort_on_warning(open3d_torch_ops)

# Set output directory according to architecture (cpu/cuda)
get_target_property(TORCH_OPS_DIR open3d_torch_ops LIBRARY_OUTPUT_DIRECTORY)
set(TORCH_OPS_ARCH_DIR
    "${TORCH_OPS_DIR}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(open3d_torch_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TORCH_OPS_ARCH_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TORCH_OPS_ARCH_DIR}")

# Do not add "lib" prefix
set_target_properties(open3d_torch_ops PROPERTIES PREFIX "")
set_target_properties(open3d_torch_ops PROPERTIES DEBUG_POSTFIX "_debug")

set( open3d_torch_ops_DEFINITIONS )
if(BUILD_CUDA_MODULE)
    target_include_directories(open3d_torch_ops SYSTEM PRIVATE
        ${CUB_INCLUDE_DIR}
        ${CUTLASS_INCLUDE_DIR}
    )
    list(APPEND open3d_torch_ops_DEFINITIONS "BUILD_CUDA_MODULE")
endif()
target_compile_definitions(open3d_torch_ops PRIVATE "${open3d_torch_ops_DEFINITIONS}")

target_include_directories(open3d_torch_ops SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(open3d_torch_ops PRIVATE
    3rdparty_tbb
    torch_cpu
    ${EIGEN3_TARGET}
    ${FMT_TARGET}
)
if( BUILD_CUDA_MODULE )
    target_link_libraries(open3d_torch_ops PRIVATE
        ${TORCH_LIBRARIES}
    )
endif()

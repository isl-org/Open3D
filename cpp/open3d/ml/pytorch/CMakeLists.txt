message(STATUS "Building PyTorch ops")

if(BUILD_CUDA_MODULE)
    message(STATUS "Building PyTorch ops with CUDA")
endif()

find_package(Pytorch REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(BUILD_CUDA_MODULE)
    find_package(CUB REQUIRED)
    find_package(CUTLASS REQUIRED)
endif()


add_library(open3d_torch_ops SHARED)

target_sources(open3d_torch_ops PRIVATE
    continuous_conv/ContinuousConvBackpropFilterOpKernel.cpp
    continuous_conv/ContinuousConvOpKernel.cpp
    continuous_conv/ContinuousConvOps.cpp
    continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cpp
    continuous_conv/ContinuousConvTransposeOpKernel.cpp
    continuous_conv/ContinuousConvTransposeOps.cpp
)

target_sources(open3d_torch_ops PRIVATE
    misc/BuildSpatialHashTableOpKernel.cpp
    misc/BuildSpatialHashTableOps.cpp
    misc/FixedRadiusSearchOpKernel.cpp
    misc/FixedRadiusSearchOps.cpp
    misc/InvertNeighborsListOpKernel.cpp
    misc/InvertNeighborsListOps.cpp
    misc/KnnSearchOpKernel.cpp
    misc/KnnSearchOps.cpp
    misc/NmsOps.cpp
    misc/RadiusSearchOpKernel.cpp
    misc/RadiusSearchOps.cpp
    misc/RaggedToDenseOpKernel.cpp
    misc/RaggedToDenseOps.cpp
    misc/ReduceSubarraysSumOpKernel.cpp
    misc/ReduceSubarraysSumOps.cpp
    misc/RoiPoolOps.cpp
    misc/VoxelizeOpKernel.cpp
    misc/VoxelizeOps.cpp
    misc/VoxelPoolingOpKernel.cpp
    misc/VoxelPoolingOps.cpp
)

target_sources(open3d_torch_ops PRIVATE
ragged_tensor/RaggedTensor.cpp)

target_sources(open3d_torch_ops PRIVATE
    pointnet/BallQueryOps.cpp
    pointnet/InterpolateOps.cpp
    pointnet/SamplingOps.cpp
)

target_sources(open3d_torch_ops PRIVATE
    ../contrib/Nms.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(open3d_torch_ops PRIVATE
        continuous_conv/ContinuousConvBackpropFilterOpKernel.cu
        continuous_conv/ContinuousConvOpKernel.cu
        continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cu
        continuous_conv/ContinuousConvTransposeOpKernel.cu
    )

    target_sources(open3d_torch_ops PRIVATE
        misc/BuildSpatialHashTableOpKernel.cu
        misc/FixedRadiusSearchOpKernel.cu
        misc/InvertNeighborsListOpKernel.cu
        misc/RaggedToDenseOpKernel.cu
        misc/ReduceSubarraysSumOpKernel.cu
        misc/VoxelizeOpKernel.cu
    )

    target_sources(open3d_torch_ops PRIVATE
        pointnet/BallQueryKernel.cu
        pointnet/InterpolateKernel.cu
        pointnet/SamplingKernel.cu
    )

    target_sources(open3d_torch_ops PRIVATE
        ../impl/continuous_conv/ContinuousConvCUDAKernels.cu
    )

    target_sources(open3d_torch_ops PRIVATE
        ../contrib/BallQuery.cu
        ../contrib/InterpolatePoints.cu
        ../contrib/Nms.cu
        ../contrib/RoiPoolKernel.cu
    )
endif()

open3d_show_and_abort_on_warning(open3d_torch_ops)
open3d_set_global_properties(open3d_torch_ops)

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

if (BUILD_CUDA_MODULE)
    target_include_directories(open3d_torch_ops SYSTEM PRIVATE
        ${CUB_INCLUDE_DIR}
        ${CUTLASS_INCLUDE_DIR}
    )
endif()

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
    Open3D::Open3D
)

if (BUILD_CUDA_MODULE)
    target_link_libraries(open3d_torch_ops PRIVATE
        ${TORCH_LIBRARIES}
    )
endif()

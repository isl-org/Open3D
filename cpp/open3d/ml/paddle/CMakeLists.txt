if(BUILD_CUDA_MODULE)
    message(STATUS "Building Paddle ops with CUDA")
else()
    message(STATUS "Building Paddle ops")
endif()

set(PRINT_ONCE ON)
find_package(Paddle REQUIRED)

if (Paddle_VERSION VERSION_LESS 3.0.0)
    message(FATAL_ERROR "Please update to Paddle 3.0.0+ to build Paddle Ops.")
endif()

add_library(open3d_paddle_ops SHARED)

target_sources(open3d_paddle_ops PRIVATE
    PaddleHelper.cpp
    misc/BuildSpatialHashTableOpKernel.cpp
    misc/BuildSpatialHashTableOps.cpp
    misc/FixedRadiusSearchOps.cpp
    misc/FixedRadiusSearchOpKernel.cpp
    misc/RadiusSearchOps.cpp
    misc/RadiusSearchOpKernel.cpp
    misc/InvertNeighborsListOps.cpp
    misc/InvertNeighborsListOpKernel.cpp
    misc/KnnSearchOps.cpp
    misc/KnnSearchOpKernel.cpp
    misc/RaggedToDenseOpKernel.cpp
    misc/RaggedToDenseOps.cpp
    misc/NmsOps.cpp
    misc/VoxelizeOpKernel.cpp
    misc/VoxelizeOps.cpp
    misc/ReduceSubarraysSumOpKernel.cpp
    misc/ReduceSubarraysSumOps.cpp
    misc/VoxelPoolingOps.cpp
    misc/VoxelPoolingOpKernel.cpp
    misc/RoiPoolOps.cpp
)

target_sources(open3d_paddle_ops PRIVATE
    pointnet/BallQueryOps.cpp
    pointnet/InterpolateOps.cpp
    pointnet/SamplingOps.cpp
)

target_sources(open3d_paddle_ops PRIVATE
    continuous_conv/ContinuousConvOps.cpp
    continuous_conv/ContinuousConvOpKernel.cpp
    continuous_conv/ContinuousConvBackpropFilterOpKernel.cpp
    continuous_conv/ContinuousConvTransposeOps.cpp
    continuous_conv/ContinuousConvTransposeOpKernel.cpp
    continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cpp
)

target_sources(open3d_paddle_ops PRIVATE
    sparse_conv/SparseConvBackpropFilterOpKernel.cpp
    sparse_conv/SparseConvOpKernel.cpp
    sparse_conv/SparseConvOps.cpp
    sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cpp
    sparse_conv/SparseConvTransposeOpKernel.cpp
    sparse_conv/SparseConvTransposeOps.cpp
)

target_sources(open3d_paddle_ops PRIVATE
    ../contrib/Nms.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(open3d_paddle_ops PRIVATE
        misc/BuildSpatialHashTableOpKernel.cu
        misc/FixedRadiusSearchOpKernel.cu
        misc/InvertNeighborsListOpKernel.cu
        misc/RaggedToDenseOpKernel.cu
        misc/ReduceSubarraysSumOpKernel.cu
        misc/VoxelizeOpKernel.cu
    )

    target_sources(open3d_paddle_ops PRIVATE
        pointnet/BallQueryKernel.cu
        pointnet/InterpolateKernel.cu
        pointnet/SamplingKernel.cu
    )

    target_sources(open3d_paddle_ops PRIVATE
        continuous_conv/ContinuousConvOpKernel.cu
        continuous_conv/ContinuousConvBackpropFilterOpKernel.cu
        continuous_conv/ContinuousConvTransposeOpKernel.cu
        continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cu
    )
    target_sources(open3d_paddle_ops PRIVATE
        sparse_conv/SparseConvBackpropFilterOpKernel.cu
        sparse_conv/SparseConvOpKernel.cu
        sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cu
        sparse_conv/SparseConvTransposeOpKernel.cu
    )
    target_sources(open3d_paddle_ops PRIVATE
        ../contrib/BallQuery.cu
        ../contrib/InterpolatePoints.cu
        ../contrib/Nms.cu
        ../contrib/RoiPoolKernel.cu
    )

    target_sources(open3d_paddle_ops PRIVATE
        ../impl/continuous_conv/ContinuousConvCUDAKernels.cu
        ../impl/sparse_conv/SparseConvCUDAKernels.cu
    )
endif()

open3d_show_and_abort_on_warning(open3d_paddle_ops)
open3d_set_global_properties(open3d_paddle_ops)

# Set output directory according to architecture (cpu/cuda)
get_target_property(PADDLE_OPS_DIR open3d_paddle_ops LIBRARY_OUTPUT_DIRECTORY)
set(PADDLE_OPS_ARCH_DIR
    "${PADDLE_OPS_DIR}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(open3d_paddle_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PADDLE_OPS_ARCH_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${PADDLE_OPS_ARCH_DIR}")

# Do not add "lib" prefix
set_target_properties(open3d_paddle_ops PROPERTIES PREFIX "")
set_target_properties(open3d_paddle_ops PROPERTIES DEBUG_POSTFIX "_debug")

target_include_directories(open3d_paddle_ops SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${PADDLE_INCLUDE_DIRS}
)

target_link_libraries(open3d_paddle_ops PRIVATE
    paddle
    Open3D::Open3D
    Open3D::3rdparty_eigen3
    Open3D::3rdparty_fmt
    Open3D::3rdparty_nanoflann
    Open3D::3rdparty_parallelstl
    Open3D::3rdparty_tbb
)

if (BUILD_CUDA_MODULE)
    target_link_libraries(open3d_paddle_ops PRIVATE
        Open3D::3rdparty_cutlass
        ${PADDLE_LIBRARIES}
        CUDA::cuda_driver
    )

    if (TARGET Open3D::3rdparty_cub)
        target_link_libraries(open3d_paddle_ops PRIVATE
            Open3D::3rdparty_cub
        )
    endif()
endif()

install(TARGETS open3d_paddle_ops EXPORT Open3DPaddleOps
    LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
)
install(EXPORT Open3DPaddleOps NAMESPACE ${PROJECT_NAME}:: DESTINATION ${Open3D_INSTALL_CMAKE_DIR})

if (BUILD_SHARED_LIBS AND UNIX)
file(CONFIGURE OUTPUT open3d_paddle_ops.pc.in
         CONTENT [=[
prefix=${pcfiledir}/../..
libdir=${prefix}/lib
includedir=${prefix}/include/

Name: Open3D Paddle Ops
Description: @PROJECT_DESCRIPTION@ This library contains 3D ML Ops for use with Paddle.
URL: @PROJECT_HOMEPAGE_URL@
Version: @PROJECT_VERSION@
Requires: Open3D = @PROJECT_VERSION@
Cflags:
Libs: -lopen3d_paddle_ops]=]  @ONLY NEWLINE_STYLE LF)
    file(GENERATE OUTPUT open3d_paddle_ops.pc INPUT
        "${CMAKE_CURRENT_BINARY_DIR}/open3d_paddle_ops.pc.in"
        TARGET open3d_paddle_ops)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/open3d_paddle_ops.pc"
        DESTINATION "${Open3D_INSTALL_LIB_DIR}/pkgconfig")
endif()

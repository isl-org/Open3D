message(STATUS "Building TensorFlow ops")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(BUILD_CUDA_MODULE)
    message(STATUS "Building TensorFlow ops with CUDA")
endif()

find_package(Tensorflow REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(BUILD_CUDA_MODULE)
    find_package(CUB REQUIRED)
    find_package(CUTLASS REQUIRED)
endif()


add_library(open3d_tf_ops SHARED)

target_sources(open3d_tf_ops PRIVATE
    continuous_conv/ContinuousConvBackpropFilterOpKernel.cpp
    continuous_conv/ContinuousConvBackpropFilterOps.cpp
    continuous_conv/ContinuousConvOpKernel.cpp
    continuous_conv/ContinuousConvOps.cpp
    continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cpp
    continuous_conv/ContinuousConvTransposeBackpropFilterOps.cpp
    continuous_conv/ContinuousConvTransposeOpKernel.cpp
    continuous_conv/ContinuousConvTransposeOps.cpp
)

target_sources(open3d_tf_ops PRIVATE
    sparse_conv/SparseConvBackpropFilterOpKernel.cpp
    sparse_conv/SparseConvOpKernel.cpp
    sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cpp
    sparse_conv/SparseConvTransposeOpKernel.cpp
    sparse_conv/SparseConvBackpropFilterOps.cpp
    sparse_conv/SparseConvOps.cpp
    sparse_conv/SparseConvTransposeBackpropFilterOps.cpp
    sparse_conv/SparseConvTransposeOps.cpp
)

target_sources(open3d_tf_ops PRIVATE
    misc/BuildSpatialHashTableOpKernel.cpp
    misc/BuildSpatialHashTableOps.cpp
    misc/FixedRadiusSearchOpKernel.cpp
    misc/FixedRadiusSearchOps.cpp
    misc/InvertNeighborsListOpKernel.cpp
    misc/InvertNeighborsListOps.cpp
    misc/KnnSearchOpKernel.cpp
    misc/KnnSearchOps.cpp
    misc/NmsOpKernel.cpp
    misc/NmsOps.cpp
    misc/RadiusSearchOpKernel.cpp
    misc/RadiusSearchOps.cpp
    misc/ReduceSubarraysSumOpKernel.cpp
    misc/ReduceSubarraysSumOps.cpp
    misc/VoxelizeOpKernel.cpp
    misc/VoxelizeOps.cpp
    misc/VoxelPoolingGradOpKernel.cpp
    misc/VoxelPoolingOpKernel.cpp
    misc/VoxelPoolingOps.cpp
)

target_sources(open3d_tf_ops PRIVATE
    pointnet/BallQueryOps.cpp
    pointnet/InterpolateOps.cpp
    pointnet/RoiPoolOps.cpp
    pointnet/SamplingOps.cpp
)

target_sources(open3d_tf_ops PRIVATE
    tf_neighbors/tf_batch_neighbors.cpp
    tf_neighbors/tf_neighbors.cpp
)

target_sources(open3d_tf_ops PRIVATE
    tf_subsampling/tf_batch_subsampling.cpp
    tf_subsampling/tf_subsampling.cpp
)

target_sources(open3d_tf_ops PRIVATE
    ../contrib/Cloud.cpp
    ../contrib/GridSubsampling.cpp
    ../contrib/neighbors.cpp
    ../contrib/Nms.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(open3d_tf_ops PRIVATE
        continuous_conv/ContinuousConvBackpropFilterOpKernel.cu
        continuous_conv/ContinuousConvOpKernel.cu
        continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cu
        continuous_conv/ContinuousConvTransposeOpKernel.cu
    )

    target_sources(open3d_tf_ops PRIVATE
        sparse_conv/SparseConvBackpropFilterOpKernel.cu
        sparse_conv/SparseConvOpKernel.cu
        sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cu
        sparse_conv/SparseConvTransposeOpKernel.cu
    )

    target_sources(open3d_tf_ops PRIVATE
        misc/BuildSpatialHashTableOpKernel.cu
        misc/FixedRadiusSearchOpKernel.cu
        misc/InvertNeighborsListOpKernel.cu
        misc/NmsOpKernel.cu
        misc/ReduceSubarraysSumOpKernel.cu
        misc/VoxelizeOpKernel.cu
    )

    target_sources(open3d_tf_ops PRIVATE
        pointnet/BallQueryOpKernel.cu
        pointnet/InterpolateOpKernel.cu
        pointnet/RoiPoolOpKernel.cu
        pointnet/SamplingOpKernel.cu
    )

    target_sources(open3d_tf_ops PRIVATE
        ../impl/continuous_conv/ContinuousConvCUDAKernels.cu
        ../impl/sparse_conv/SparseConvCUDAKernels.cu
    )

    target_sources(open3d_tf_ops PRIVATE
        ../contrib/BallQuery.cu
        ../contrib/InterpolatePoints.cu
        ../contrib/Nms.cu
        ../contrib/RoiPoolKernel.cu
    )
endif()

open3d_show_and_abort_on_warning(open3d_tf_ops)
# do not use open3d_set_global_properties(open3d_tf_ops) here because some
# options are not compatible for tf op libraries

# Set output directory according to architecture (cpu/cuda)
get_target_property(TF_OPS_DIR open3d_tf_ops LIBRARY_OUTPUT_DIRECTORY)
set(TF_OPS_ARCH_DIR
    "${TF_OPS_DIR}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(open3d_tf_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TF_OPS_ARCH_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TF_OPS_ARCH_DIR}")

# Do not add "lib" prefix
set_target_properties(open3d_tf_ops PROPERTIES PREFIX "")
set_target_properties(open3d_tf_ops PROPERTIES DEBUG_POSTFIX "_debug")

# enable c++11
set_target_properties(open3d_tf_ops PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_STANDARD 11
)

target_compile_definitions(open3d_tf_ops PRIVATE "${Tensorflow_DEFINITIONS}")

if (BUILD_CUDA_MODULE)
    target_include_directories(open3d_tf_ops SYSTEM PRIVATE
        ${CUB_INCLUDE_DIR}
        ${CUTLASS_INCLUDE_DIR}
    )

    target_compile_definitions(open3d_tf_ops PRIVATE "BUILD_CUDA_MODULE")
endif()

# Silence warnings caused by tensorflow's Eigen. Travis has a different version
# of protobuf in /usr/local/include TBB is in /usr/local/include, so it needs to
# be after TensorFlow
target_include_directories(open3d_tf_ops SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${Tensorflow_INCLUDE_DIR}
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(open3d_tf_ops PRIVATE
    3rdparty_tbb
    ${Tensorflow_FRAMEWORK_LIB}
    ${FMT_TARGET}
    Open3D::Open3D
)

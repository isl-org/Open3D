message(STATUS "Building Tensorflow ops")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(BUILD_CUDA_MODULE)
    message(STATUS "Building Tensorflow ops with CUDA")
endif()

find_package(Tensorflow REQUIRED)
find_package(nanoflann REQUIRED)
find_package(parallelstl REQUIRED)
if(BUILD_CUDA_MODULE)
    find_package(CUB REQUIRED)
    find_package(CUTLASS REQUIRED)
endif()

set(TF_OPS_SOURCES
    "ContinuousConv/ContinuousConvBackpropFilterOpKernel.cpp"
    "ContinuousConv/ContinuousConvOpKernel.cpp"
    "ContinuousConv/ContinuousConvTransposeBackpropFilterOpKernel.cpp"
    "ContinuousConv/ContinuousConvTransposeOpKernel.cpp"
    "ContinuousConv/ContinuousConvBackpropFilterOps.cpp"
    "ContinuousConv/ContinuousConvOps.cpp"
    "ContinuousConv/ContinuousConvTransposeBackpropFilterOps.cpp"
    "ContinuousConv/ContinuousConvTransposeOps.cpp"
    "Misc/BuildSpatialHashTableOpKernel.cpp"
    "Misc/RadiusSearchOpKernel.cpp"
    "Misc/InvertNeighborsListOps.cpp"
    "Misc/ReduceSubarraysSumOps.cpp"
    "Misc/VoxelPoolingOps.cpp"
    "Misc/FixedRadiusSearchOps.cpp"
    "Misc/KnnSearchOpKernel.cpp"
    "Misc/RadiusSearchOps.cpp"
    "Misc/VoxelPoolingGradOpKernel.cpp"
    "Misc/BuildSpatialHashTableOps.cpp"
    "Misc/InvertNeighborsListOpKernel.cpp"
    "Misc/ReduceSubarraysSumOpKernel.cpp"
    "Misc/FixedRadiusSearchOpKernel.cpp"
    "Misc/KnnSearchOps.cpp"
    "Misc/VoxelPoolingOpKernel.cpp"
    "tf_neighbors/tf_batch_neighbors.cpp"
    "tf_neighbors/tf_neighbors.cpp"
    "tf_subsampling/tf_batch_subsampling.cpp"
    "tf_subsampling/tf_subsampling.cpp"
    "../contrib/Cloud.cpp"
    "../contrib/GridSubsampling.cpp"
    "../contrib/neighbors.cpp"
)

set(TF_OPS_CUDA_SOURCES
    "ContinuousConv/ContinuousConvBackpropFilterOpKernel.cu"
    "ContinuousConv/ContinuousConvOpKernel.cu"
    "ContinuousConv/ContinuousConvTransposeBackpropFilterOpKernel.cu"
    "ContinuousConv/ContinuousConvTransposeOpKernel.cu"
    "Misc/FixedRadiusSearchOpKernel.cu"
    "Misc/BuildSpatialHashTableOpKernel.cu"
    "Misc/InvertNeighborsListOpKernel.cu"
    "Misc/ReduceSubarraysSumOpKernel.cu"
    "../impl/continuous_conv/ContinuousConvCUDAKernels.cu"
)

if(BUILD_CUDA_MODULE)
    set(open3d_tf_ops_SOURCES
        ${TF_OPS_SOURCES}
        ${TF_OPS_CUDA_SOURCES}
    )
else()
    set(open3d_tf_ops_SOURCES
        ${TF_OPS_SOURCES}
    )
endif()

add_library(open3d_tf_ops SHARED ${open3d_tf_ops_SOURCES})
# do not use open3d_set_global_properties(open3d_tf_ops) here because some
# options are not compatible for tf op libraries
open3d_show_and_abort_on_warning(open3d_tf_ops)

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
set_target_properties( open3d_tf_ops PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_STANDARD 11
        )

set( open3d_tf_ops_DEFINITIONS "${Tensorflow_DEFINITIONS}" )
if(BUILD_CUDA_MODULE)
    target_include_directories(open3d_tf_ops SYSTEM PRIVATE ${CUB_INCLUDE_DIR}
                                                     ${CUTLASS_INCLUDE_DIR})
    list(APPEND open3d_tf_ops_DEFINITIONS "BUILD_CUDA_MODULE")
endif()

target_compile_definitions(open3d_tf_ops PRIVATE "${open3d_tf_ops_DEFINITIONS}")

# Silence warnings caused by tensorflow's Eigen. Travis has a different version
# of protobuf in /usr/local/include TBB is in /usr/local/include, so it needs to
# be after Tensorflow
target_include_directories(open3d_tf_ops SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${Tensorflow_INCLUDE_DIR}
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
)

target_link_libraries(open3d_tf_ops PRIVATE
    3rdparty_tbb
    ${Tensorflow_FRAMEWORK_LIB}
    ${FMT_TARGET}
)

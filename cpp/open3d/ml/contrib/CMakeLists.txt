set(OPEN3D_ML_CONTRIB_CUDA_SOURCES
    IoU.cu
)

add_library(ml_contrib OBJECT)

target_sources(ml_contrib PRIVATE
    Cloud.cpp
    GridSubsampling.cpp
    IoU.cpp
)

open3d_add_split_module_sources(
    HOST_TARGET ml_contrib
    DEVICE_OBJECT ml_contrib_cuda
    DEVICE_KIND CUDA
    MODULE_FLAG BUILD_CUDA_MODULE
    PLAIN_OBJECT
    SOURCES ${OPEN3D_ML_CONTRIB_CUDA_SOURCES}
)


open3d_show_and_abort_on_warning(ml_contrib)
open3d_set_global_properties(ml_contrib)
open3d_set_open3d_lib_properties(ml_contrib)
open3d_link_3rdparty_libraries(ml_contrib)


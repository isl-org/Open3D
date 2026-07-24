add_library(ml_contrib OBJECT)

target_sources(ml_contrib PRIVATE
    Cloud.cpp
    GridSubsampling.cpp
)

if(BUILD_CUDA_MODULE)
    target_sources(ml_contrib PRIVATE
        IoU.cu
    )
endif()

if(BUILD_SYCL_MODULE)
    # IoU.cpp only defines CPU kernels; IoUSYCL.cpp holds SYCL device code.
    open3d_sycl_target_sources(ml_contrib PRIVATE
        IoUSYCL.cpp
    )
    target_sources(ml_contrib PRIVATE
        IoU.cpp
    )
else()
    target_sources(ml_contrib PRIVATE
        IoU.cpp
    )
endif()

open3d_show_and_abort_on_warning(ml_contrib)
open3d_set_global_properties(ml_contrib)
open3d_set_open3d_lib_properties(ml_contrib)
open3d_link_3rdparty_libraries(ml_contrib)

if(BUILD_CUDA_MODULE)
    target_include_directories(ml_contrib SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

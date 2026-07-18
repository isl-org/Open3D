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
    # IoU.cpp is compiled SYCL-aware (-fsycl) too: IoU.h declares
    # IoU{Bev,3d}SYCLKernel(sycl::queue&, ...) under BUILD_SYCL_MODULE, which
    # needs the full SYCL runtime even though IoU.cpp itself only defines the
    # CPU kernels.
    open3d_sycl_target_sources(ml_contrib PRIVATE
        IoU.cpp
        IoUSYCL.cpp
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

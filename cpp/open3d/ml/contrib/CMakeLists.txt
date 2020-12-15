set (SRC
    Cloud.cpp
    GridSubsampling.cpp
    contrib_nns.cpp
    IoU.cpp
)

if(BUILD_CUDA_MODULE)
    set (SRC
        ${SRC}
        IoU.cu
    )
endif()

add_library(ml_contrib OBJECT ${SRC})
open3d_set_global_properties(ml_contrib)
open3d_link_3rdparty_libraries(ml_contrib)
open3d_set_open3d_lib_properties(ml_contrib)
open3d_show_and_abort_on_warning(ml_contrib)

if(BUILD_CUDA_MODULE)
    target_include_directories(ml_contrib SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_library(Open3D)

open3d_sycl_target_sources(Open3D PRIVATE VERBOSE
    SYCLUtils.cpp
)

open3d_show_and_abort_on_warning(Open3D)
open3d_set_global_properties(Open3D)
open3d_link_3rdparty_libraries(Open3D)

include(CMakePackageConfigHelpers)





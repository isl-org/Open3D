open3d_ispc_add_library(camera OBJECT)

target_sources(camera PRIVATE
    PinholeCameraIntrinsic.cpp
    PinholeCameraParameters.cpp
    PinholeCameraTrajectory.cpp
)

open3d_show_and_abort_on_warning(camera)
open3d_set_global_properties(camera)
open3d_set_open3d_lib_properties(camera)
open3d_link_3rdparty_libraries(camera)

# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Odometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(Odometry)
open3d_set_global_properties(Odometry)
open3d_link_3rdparty_libraries(Odometry)

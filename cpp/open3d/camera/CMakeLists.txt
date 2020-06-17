# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Camera OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(Camera)
open3d_set_global_properties(Camera)
open3d_link_3rdparty_libraries(Camera)

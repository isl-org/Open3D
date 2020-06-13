# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Registration OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(Registration)
open3d_set_global_properties(Registration)
open3d_link_3rdparty_libraries(Registration)

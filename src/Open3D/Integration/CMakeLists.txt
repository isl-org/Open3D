# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Integration OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(Integration)
open3d_set_global_properties(Integration)
open3d_link_3rdparty_libraries(Integration)

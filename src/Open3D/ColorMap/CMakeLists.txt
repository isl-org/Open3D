# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(ColorMap OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(ColorMap)
open3d_set_global_properties(ColorMap)
open3d_link_3rdparty_libraries(ColorMap)

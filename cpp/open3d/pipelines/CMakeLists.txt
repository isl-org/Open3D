file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

add_library(pipelines OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(pipelines)
open3d_set_global_properties(pipelines)
open3d_link_3rdparty_libraries(pipelines)

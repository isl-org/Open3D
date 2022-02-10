open3d_ispc_add_library(data OBJECT)

target_sources(data PRIVATE
    Dataset.cpp
)

open3d_show_and_abort_on_warning(data)
open3d_set_global_properties(data)
open3d_set_open3d_lib_properties(data)
open3d_link_3rdparty_libraries(data)

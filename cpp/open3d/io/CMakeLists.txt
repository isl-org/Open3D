open3d_ispc_add_library(io OBJECT)

target_sources(io PRIVATE
    NumpyIO.cpp
    HashMapIO.cpp
)

open3d_show_and_abort_on_warning(io)
open3d_set_global_properties(io)
open3d_set_open3d_lib_properties(io)
open3d_link_3rdparty_libraries(io)

open3d_ispc_add_library(tio OBJECT)

target_sources(tio PRIVATE
    NumpyIO.cpp
    HashMapIO.cpp
)

open3d_show_and_abort_on_warning(tio)
open3d_set_global_properties(tio)
open3d_set_open3d_lib_properties(tio)
open3d_link_3rdparty_libraries(tio)

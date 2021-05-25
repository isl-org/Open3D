add_library(utility OBJECT)

target_sources(utility PRIVATE
    Console.cpp
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    Timer.cpp
)

open3d_show_and_abort_on_warning(utility)
open3d_set_global_properties(utility)
open3d_set_open3d_lib_properties(utility)
open3d_link_3rdparty_libraries(utility)

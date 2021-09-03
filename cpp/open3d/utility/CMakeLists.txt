open3d_ispc_add_library(utility OBJECT)

target_sources(utility PRIVATE
    Console.cpp
    CPUInfo.cpp
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    Logging.cpp
    Parallel.cpp
    Timer.cpp
)

open3d_show_and_abort_on_warning(utility)
open3d_set_global_properties(utility)
open3d_set_open3d_lib_properties(utility)
open3d_link_3rdparty_libraries(utility)

# Due to several ISPC language emulation limitations,
# ISPC files must be added after all properties are set
# and in the same directory where the target is created.
if (BUILD_ISPC_MODULE)
    open3d_ispc_target_sources(utility PRIVATE
        ISA.cpp
        ISA.ispc
    )
endif()

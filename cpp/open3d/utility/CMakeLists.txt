open3d_ispc_add_library(utility OBJECT)

target_sources(utility PRIVATE
    CompilerInfo.cpp
    Console.cpp
    CPUInfo.cpp
    Download.cpp
    Eigen.cpp
    Extract.cpp
    ExtractZIP.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    ISAInfo.cpp
    Logging.cpp
    Parallel.cpp
    ProgressBar.cpp
    Random.cpp
    Timer.cpp
)

if (BUILD_ISPC_MODULE)
    target_sources(utility PRIVATE
        ISAInfo.ispc
    )
endif()

open3d_show_and_abort_on_warning(utility)
open3d_set_global_properties(utility)
open3d_set_open3d_lib_properties(utility)
open3d_link_3rdparty_libraries(utility)

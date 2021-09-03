target_sources(tests PRIVATE
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    Logging.cpp
    Timer.cpp
)

if (BUILD_ISPC_MODULE)
    target_sources(tests PRIVATE
        ISA.cpp
    )
endif()

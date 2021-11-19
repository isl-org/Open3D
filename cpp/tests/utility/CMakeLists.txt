target_sources(tests PRIVATE
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    ISAInfo.cpp
    Logging.cpp
    Preprocessor.cpp
    Timer.cpp
)

if (BUILD_ISPC_MODULE)
    target_sources(tests PRIVATE
        Helper.ispc
    )
endif()

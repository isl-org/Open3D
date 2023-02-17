target_sources(tests PRIVATE
    Download.cpp
    Extract.cpp
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    ISAInfo.cpp
    Logging.cpp
    Preprocessor.cpp
    ProgressBar.cpp
    Timer.cpp
    Random.cpp
)

if (BUILD_ISPC_MODULE)
    target_sources(tests PRIVATE
        Helper.ispc
    )
endif()

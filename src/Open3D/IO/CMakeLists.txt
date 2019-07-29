# Build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE IO_ALL_HEADER_FILES "*.h")
file(GLOB         RPLY_SOURCE_FILES "../../../3rdparty/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../../3rdparty/liblzf/*.c")

# Create object library
add_library(IO OBJECT
            ${IO_ALL_HEADER_FILES}
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

ShowAndAbortOnWarning(IO)

if (BUILD_AZURE_KINECT)
    find_package(k4a REQUIRED)
    find_package(k4arecord REQUIRED)
    # libk4a* must be installed separately
    target_link_libraries(IO k4a::k4a k4a::k4arecord)
endif ()

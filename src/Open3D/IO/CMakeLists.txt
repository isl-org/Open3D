# Build
file(GLOB_RECURSE CLASS_IO_SOURCE_FILES "ClassIO/*.cpp")
file(GLOB_RECURSE FILE_FORMAT_SOURCE_FILES "FileFormat/*.cpp")
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

file(GLOB         RPLY_SOURCE_FILES "../../../3rdparty/rply/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../../3rdparty/liblzf/liblzf/*.c")

if (BUILD_AZURE_KINECT)
    file(GLOB_RECURSE SENSOR_SOURCE_FILES "Sensor/*.cpp")
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${SENSOR_SOURCE_FILES})
endif ()

# Create object library
add_library(IO OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
ShowAndAbortOnWarning(IO)

if (UNIX)
    set_source_files_properties( "FileFormat/FileSTL.cpp" PROPERTIES COMPILE_FLAGS -Wno-unused-result )
endif ()

# Enforce 3rd party dependencies
add_dependencies(IO build_all_3rd_party_libs)

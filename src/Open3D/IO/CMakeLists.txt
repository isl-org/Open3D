# Build
file(GLOB_RECURSE CLASS_IO_SOURCE_FILES "ClassIO/*.cpp")
file(GLOB_RECURSE FILE_FORMAT_SOURCE_FILES "FileFormat/*.cpp")
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

if (BUILD_AZURE_KINECT)
    file(GLOB_RECURSE SENSOR_SOURCE_FILES "Sensor/*.cpp")
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${SENSOR_SOURCE_FILES})
endif ()

# Create object library
add_library(IO OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
open3d_show_and_abort_on_warning(IO)
open3d_set_global_properties(IO)
open3d_link_3rdparty_libraries(IO)

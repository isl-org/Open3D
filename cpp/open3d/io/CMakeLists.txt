# Build
file(GLOB CLASS_IO_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE FILE_FORMAT_SOURCE_FILES "file_format/*.cpp")
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

if (BUILD_AZURE_KINECT)
    file(GLOB_RECURSE SENSOR_SOURCE_FILES "sensor/*.cpp")
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${SENSOR_SOURCE_FILES})
endif ()

# Create object library
add_library(io OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
open3d_show_and_abort_on_warning(io)
open3d_set_global_properties(io)
open3d_link_3rdparty_libraries(io)

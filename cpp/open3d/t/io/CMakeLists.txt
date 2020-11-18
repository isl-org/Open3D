# Build
set(FILE_IO_SRC
    PointCloudIO.cpp
    file_format/FileXYZI.cpp
    file_format/FilePLY.cpp
)

file(GLOB SENSOR_IO_SRC "sensor/*.cpp")
if (BUILD_LIBREALSENSE)
    file(GLOB REALSENSE_SRC "sensor/realsense/*.cpp")
endif ()
if (BUILD_AZURE_KINECT)
    file(GLOB KINECT_SRC "sensor/azure_kinect/*.cpp")
endif ()

# Create object library
add_library(tio OBJECT
    ${FILE_IO_SRC}
    ${RPLY_SOURCE_FILES}
    ${LIBLZF_SOURCE_FILES}
    ${SENSOR_IO_SRC}
    ${REALSENSE_SRC}
    ${KINECT_SRC}
    )
open3d_show_and_abort_on_warning(tio)
open3d_set_global_properties(tio)
open3d_set_open3d_lib_properties(tio)
open3d_link_3rdparty_libraries(tio)

open3d_ispc_add_library(tio OBJECT)

target_sources(tio PRIVATE
    ImageIO.cpp
    NumpyIO.cpp
    HashMapIO.cpp
    PointCloudIO.cpp
    TriangleMeshIO.cpp
)

target_sources(tio PRIVATE
    file_format/FileJPG.cpp
    file_format/FilePCD.cpp
    file_format/FilePLY.cpp
    file_format/FilePNG.cpp
    file_format/FilePTS.cpp
    file_format/FileXYZI.cpp
)

target_sources(tio PRIVATE
    sensor/RGBDVideoMetadata.cpp
    sensor/RGBDVideoReader.cpp
)

if (BUILD_LIBREALSENSE)
    target_sources(tio PRIVATE
        sensor/realsense/RealSenseSensor.cpp
        sensor/realsense/RealSenseSensorConfig.cpp
        sensor/realsense/RSBagReader.cpp
    )
endif()

open3d_show_and_abort_on_warning(tio)
open3d_set_global_properties(tio)
open3d_set_open3d_lib_properties(tio)
open3d_link_3rdparty_libraries(tio)

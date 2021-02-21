# Build
set(CLASS_IO_SOURCE_FILES
    FeatureIO.cpp
    FileFormatIO.cpp
    IJsonConvertibleIO.cpp
    ImageIO.cpp
    ImageWarpingFieldIO.cpp
    LineSetIO.cpp
    ModelIO.cpp
    OctreeIO.cpp
    PinholeCameraTrajectoryIO.cpp
    PointCloudIO.cpp
    PoseGraphIO.cpp
    TriangleMeshIO.cpp
    VoxelGridIO.cpp
    )
set(FILE_FORMAT_SOURCE_FILES
    file_format/FileASSIMP.cpp
    file_format/FileBIN.cpp
    file_format/FileGLTF.cpp
    file_format/FileJPG.cpp
    file_format/FileJSON.cpp
    file_format/FileLOG.cpp
    file_format/FileOBJ.cpp
    file_format/FileOFF.cpp
    file_format/FilePCD.cpp
    file_format/FilePLY.cpp
    file_format/FilePNG.cpp
    file_format/FilePTS.cpp
    file_format/FileSTL.cpp
    file_format/FileTUM.cpp
    file_format/FileXYZ.cpp
    file_format/FileXYZN.cpp
    file_format/FileXYZRGB.cpp
    )
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

if (BUILD_AZURE_KINECT)
    set(SENSOR_SOURCE_FILES
        sensor/azure_kinect/AzureKinectRecorder.cpp
        sensor/azure_kinect/MKVMetadata.cpp
        sensor/azure_kinect/AzureKinectSensorConfig.cpp
        sensor/azure_kinect/MKVReader.cpp
        sensor/azure_kinect/AzureKinectSensor.cpp
        sensor/azure_kinect/MKVWriter.cpp
        sensor/azure_kinect/K4aPlugin.cpp
        )
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${SENSOR_SOURCE_FILES})
endif ()

if (BUILD_RPC_INTERFACE)
    set(RPC_SOURCE_FILES
        rpc/BufferConnection.cpp
        rpc/Connection.cpp
        rpc/DummyReceiver.cpp
        rpc/MessageUtils.cpp
        rpc/ReceiverBase.cpp
        rpc/RemoteFunctions.cpp
        rpc/ZMQContext.cpp
        )
    set(IO_ALL_SOURCE_FILES ${IO_ALL_SOURCE_FILES} ${RPC_SOURCE_FILES})
endif()

# Create object library
add_library(io OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
open3d_show_and_abort_on_warning(io)
open3d_set_global_properties(io)
open3d_set_open3d_lib_properties(io)
open3d_link_3rdparty_libraries(io)

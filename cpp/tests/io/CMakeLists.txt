target_sources(tests PRIVATE
    FeatureIO.cpp
    IJsonConvertibleIO.cpp
    ImageIO.cpp
    OctreeIO.cpp
    PinholeCameraTrajectoryIO.cpp
    PointCloudIO.cpp
    PoseGraphIO.cpp
    TriangleMeshIO.cpp
    VoxelGridIO.cpp
)

target_sources(tests PRIVATE
    file_format/FileBIN.cpp
    file_format/FileGLTF.cpp
    file_format/FileJPG.cpp
    file_format/FileJSON.cpp
    file_format/FileLOG.cpp
    file_format/FilePCD.cpp
    file_format/FilePLY.cpp
    file_format/FilePNG.cpp
    file_format/FilePTS.cpp
    file_format/FileSTL.cpp
    file_format/FileXYZ.cpp
    file_format/FileXYZN.cpp
    file_format/FileXYZRGB.cpp
)

target_sources(tests PRIVATE
    rpc/RemoteFunctions.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(tests PRIVATE
        sensor/AzureKinect/AzureKinectSensorConfig.cpp
    )
endif()

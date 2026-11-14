target_sources(tests PRIVATE
    FileFormatIO.cpp
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

if(BUILD_RPC_INTERFACE)
    target_sources(tests PRIVATE
        rpc/RemoteFunctions.cpp
    )
endif()

if (BUILD_AZURE_KINECT)
    target_sources(tests PRIVATE
        sensor/AzureKinect/AzureKinectSensorConfig.cpp
    )
endif()

# The USD importer is only compiled into Open3D's from-source Assimp
# (assimp.cmake sets ASSIMP_BUILD_USD_IMPORTER=ON). System/vcpkg Assimp
# packages (used e.g. on Windows ARM64) generally omit the USD importer, so
# the USD read test is only meaningful when Assimp is built from source.
if(NOT USE_SYSTEM_ASSIMP)
    set_source_files_properties(FileFormatIO.cpp PROPERTIES
        COMPILE_DEFINITIONS OPEN3D_ASSIMP_HAS_USD=1)
endif()

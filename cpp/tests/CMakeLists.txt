
cmake_minimum_required(VERSION 3.0)
# Overide policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

include_directories(".")

file(GLOB_RECURSE UNIT_TEST_SOURCE_FILES "*.cpp")
set(UNIT_TEST_SOURCE_FILES
    Main.cpp
    camera/PinholeCameraTrajectory.cpp
    camera/PinholeCameraParameters.cpp
    camera/PinholeCameraIntrinsic.cpp
    core/Indexer.cpp
    core/Hashmap.cpp
    core/Linalg.cpp
    core/NearestNeighborSearch.cpp
    core/CUDAState.cpp
    core/Blob.cpp
    core/Scalar.cpp
    core/TensorList.cpp
    core/Device.cpp
    core/TensorObject.cpp
    core/NanoFlannIndex.cpp
    core/ShapeUtil.cpp
    core/MemoryManager.cpp
    core/Tensor.cpp
    core/SizeVector.cpp
    core/EigenConverter.cpp
    geometry/PointCloud.cpp
    geometry/TriangleMesh.cpp
    geometry/VoxelGrid.cpp
    geometry/TetraMesh.cpp
    geometry/Image.cpp
    geometry/KDTreeFlann.cpp
    geometry/LineSet.cpp
    geometry/RGBDImage.cpp
    geometry/IntersectionTest.cpp
    geometry/EstimateNormals.cpp
    geometry/Line3D.cpp
    geometry/Octree.cpp
    geometry/HalfEdgeTriangleMesh.cpp
    geometry/AccumulatedPoint.cpp
    io/TriangleMeshIO.cpp
    io/IJsonConvertibleIO.cpp
    io/PointCloudIO.cpp
    io/file_format/FileSTL.cpp
    io/file_format/FileJSON.cpp
    io/file_format/FileLOG.cpp
    io/file_format/FileBIN.cpp
    io/file_format/FilePCD.cpp
    io/file_format/FileJPG.cpp
    io/file_format/FilePNG.cpp
    io/file_format/FilePLY.cpp
    io/file_format/FileXYZRGB.cpp
    io/file_format/FilePTS.cpp
    io/file_format/FileGLTF.cpp
    io/file_format/FileXYZ.cpp
    io/file_format/FileXYZN.cpp
    io/PinholeCameraTrajectoryIO.cpp
    io/PoseGraphIO.cpp
    io/ImageIO.cpp
    io/OctreeIO.cpp
    io/VoxelGridIO.cpp
    io/FeatureIO.cpp
    ml/ShapeChecking.cpp
    pipelines/registration/GlobalOptimization.cpp
    pipelines/registration/Registration.cpp
    pipelines/registration/FastGlobalRegistration.cpp
    pipelines/registration/ColoredICP.cpp
    pipelines/registration/PoseGraph.cpp
    pipelines/registration/TransformationEstimation.cpp
    pipelines/registration/CorrespondenceChecker.cpp
    pipelines/registration/GlobalOptimizationConvergenceCriteria.cpp
    pipelines/registration/Feature.cpp
    pipelines/integration/UniformTSDFVolume.cpp
    pipelines/integration/ScalableTSDFVolume.cpp
    pipelines/odometry/OdometryTools.cpp
    pipelines/odometry/RGBDOdometryJacobianFromColorTerm.cpp
    pipelines/odometry/RGBDOdometryJacobianFromHybridTerm.cpp
    pipelines/odometry/Odometry.cpp
    pipelines/odometry/OdometryOption.cpp
    t/io/PointCloudIO.cpp
    t/io/ImageIO.cpp
    t/pipelines/odometry/RGBDOdometry.cpp
    t/pipelines/registration/Registration.cpp
    t/pipelines/registration/TransformationEstimation.cpp
    t/pipelines/TransformationConverter.cpp
    t/geometry/PointCloud.cpp
    t/geometry/TriangleMesh.cpp
    t/geometry/Image.cpp
    t/geometry/TensorMap.cpp
    t/geometry/TSDFVoxelGrid.cpp
    test_utility/Compare.cpp
    test_utility/Sort.cpp
    test_utility/Raw.cpp
    test_utility/Rand.cpp
    utility/Console.cpp
    utility/Timer.cpp
    utility/Helper.cpp
    utility/FileSystem.cpp
    utility/Eigen.cpp
    utility/IJsonConvertible.cpp
    )

if (BUILD_AZURE_KINECT)
    list(APPEND UNIT_TEST_SOURCE_FILES
        io/sensor/AzureKinect/AzureKinectSensorConfig.cpp
        )
endif()

if (BUILD_GUI)
    list(APPEND UNIT_TEST_SOURCE_FILES
        visualization/rendering/MaterialModifier.cpp
    )
endif()

if (BUILD_RPC_INTERFACE)
    list(APPEND UNIT_TEST_SOURCE_FILES io/rpc/RemoteFunctions.cpp)
endif()

if (BUILD_CUDA_MODULE)
    list(APPEND UNIT_TEST_SOURCE_FILES core/FixedRadiusIndex.cpp)
endif()

if (WITH_FAISS)
    list(APPEND UNIT_TEST_SOURCE_FILES core/FaissIndex.cpp)
endif()

add_executable(tests ${UNIT_TEST_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")

# If gpu not available, add "DISABLED_" to the gpu test names
if(BUILD_CUDA_MODULE)
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=) # Empty string
else()
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=DISABLED_)
endif()

if (WITH_IPPICV)
    target_compile_definitions(tests PRIVATE WITH_IPPICV
        IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(tests PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()

find_package(Threads)

target_link_libraries(tests PRIVATE Threads::Threads ${CMAKE_PROJECT_NAME} ${JSONCPP_TARGET} ${GOOGLETEST_TARGET})
open3d_show_and_abort_on_warning(tests)
open3d_set_global_properties(tests)

if (BUILD_AZURE_KINECT)
    # K4A headres are directly used in test. Currently we don't need to link
    # the K4A libraries.
    target_include_directories(tests SYSTEM PRIVATE ${K4A_INCLUDE_DIR})
endif()

if (BUILD_CUDA_MODULE)
    # We still need to explicitly link against CUDA libraries.
    # See: https://stackoverflow.com/a/48540499/1255535.
    # Consider removing dependencies of cuda headers in the future.
    target_link_libraries(tests PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(tests SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

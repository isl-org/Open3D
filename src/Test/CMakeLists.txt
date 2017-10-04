add_executable(TestPointCloud TestPointCloud.cpp)
target_link_libraries(TestPointCloud Visualization IO Core)
set_target_properties(TestPointCloud PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestCameraPoseTrajectory TestCameraPoseTrajectory)
target_link_libraries(TestCameraPoseTrajectory Visualization IO Core)
set_target_properties(TestCameraPoseTrajectory PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestLineSet TestLineSet.cpp)
target_link_libraries(TestLineSet Visualization IO Core)
set_target_properties(TestLineSet PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestTriangleMesh TestTriangleMesh.cpp)
target_link_libraries(TestTriangleMesh Visualization IO Core)
set_target_properties(TestTriangleMesh PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestVisualizer TestVisualizer.cpp)
target_link_libraries(TestVisualizer Visualization IO Core)
set_target_properties(TestVisualizer PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestOpenMP TestOpenMP.cpp)
target_link_libraries(TestOpenMP Core)
set_target_properties(TestOpenMP PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestFlann TestFlann.cpp)
target_link_libraries(TestFlann Visualization IO Core)
set_target_properties(TestFlann PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestFileSystem TestFileSystem.cpp)
target_link_libraries(TestFileSystem IO Core)
set_target_properties(TestFileSystem PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestProgramOptions TestProgramOptions.cpp)
target_link_libraries(TestProgramOptions Core)
set_target_properties(TestProgramOptions PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestDepthCapture TestDepthCapture.cpp)
target_link_libraries(TestDepthCapture Core Visualization IO)
set_target_properties(TestDepthCapture PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestPCDFileFormat TestPCDFileFormat.cpp)
target_link_libraries(TestPCDFileFormat Core Visualization IO)
set_target_properties(TestPCDFileFormat PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestFileDialog TestFileDialog.cpp)
target_link_libraries(TestFileDialog Core IO tinyfiledialogs)
set_target_properties(TestFileDialog PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

if(Open3D_BUILD_LIBREALSENSE)
    add_executable(TestRealSense TestRealSense.cpp)
    target_link_libraries(TestRealSense Visualization realsense)
    set_target_properties(TestRealSense PROPERTIES
			FOLDER "Test"
			RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
endif(Open3D_BUILD_LIBREALSENSE)

add_executable(TestImage TestImage.cpp)
target_link_libraries(TestImage Core IO)
set_target_properties(TestImage PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestPoseGraph TestPoseGraph.cpp)
target_link_libraries(TestPoseGraph Core IO)
set_target_properties(TestPoseGraph PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

add_executable(TestRegistrationRANSAC TestRegistrationRANSAC.cpp)
target_link_libraries(TestRegistrationRANSAC Visualization Core IO)
set_target_properties(TestRegistrationRANSAC PROPERTIES
		FOLDER "Test"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")

#file(GLOB TEST_DATA_FILES "TestData/*.*")
#file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/TestData)

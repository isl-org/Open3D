macro (TEST TEST_NAME)
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TEST_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "examples/Cpp/Test"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
endmacro (TEST)

TEST(TestCameraPoseTrajectory Open3D)
TEST(TestDepthCapture         Open3D)
TEST(TestFileDialog           Open3D tinyfiledialogs)
TEST(TestFileSystem           Open3D)
TEST(TestFlann                Open3D)
TEST(TestImage                Open3D)
TEST(TestLineSet              Open3D)
TEST(TestOpenMP               Open3D)
TEST(TestPCDFileFormat        Open3D)
TEST(TestPointCloud           Open3D)
TEST(TestPoseGraph            Open3D)
TEST(TestProgramOptions       Open3D)

if (BUILD_LIBREALSENSE)
    TEST(TestRealSense        Open3D realsense)
endif (BUILD_LIBREALSENSE)

TEST(TestRGBDOdometry         Open3D)
TEST(TestRegistrationRANSAC   Open3D)
TEST(TestTriangleMesh         Open3D)
TEST(TestVisualizer           Open3D)

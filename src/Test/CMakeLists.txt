macro(TEST TEST_NAME)
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TEST_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "Test"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
endmacro(TEST)

TEST(TestCameraPoseTrajectory Core IO Visualization)
TEST(TestDepthCapture         Core IO Visualization)
TEST(TestFileDialog           Core IO tinyfiledialogs)
TEST(TestFileSystem           Core IO)
TEST(TestFlann                Core IO Visualization)
TEST(TestImage                Core IO)
TEST(TestLineSet              Core IO Visualization)
TEST(TestOpenMP               Core)
TEST(TestPCDFileFormat        Core IO Visualization)
TEST(TestPointCloud           Core IO Visualization)
TEST(TestPoseGraph            Core IO)
TEST(TestProgramOptions       Core)

if(Open3D_BUILD_LIBREALSENSE)
    TEST(TestRealSense        Visualization realsense)
endif(Open3D_BUILD_LIBREALSENSE)

TEST(TestRegistrationRANSAC   Core IO Visualization)
TEST(TestTriangleMesh         Core IO Visualization)
TEST(TestVisualizer           Core IO Visualization)

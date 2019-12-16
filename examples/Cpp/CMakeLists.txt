macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")
    ShowAndAbortOnWarning(${EXAMPLE_CPP_NAME})

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            FOLDER "examples/Cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(CameraPoseTrajectory      ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ColorMapOptimization      ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(DepthCapture              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(EvaluateFeatureMatch      ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(EvaluatePCDMatch          ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(FileDialog                ${CMAKE_PROJECT_NAME} tinyfiledialogs)
EXAMPLE_CPP(FileSystem                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Flann                     ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Image                     ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(IntegrateRGBD             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(LineSet                   ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Log                       ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(OdometryRGBD              ${CMAKE_PROJECT_NAME})
if (WITH_OPENMP)
    EXAMPLE_CPP(OpenMP                ${CMAKE_PROJECT_NAME})
endif (WITH_OPENMP)
EXAMPLE_CPP(PCDFileFormat             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(PointCloud                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(PoseGraph                 ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ProgramOptions            ${CMAKE_PROJECT_NAME})
if (BUILD_LIBREALSENSE)
    EXAMPLE_CPP(RealSense             ${CMAKE_PROJECT_NAME} realsense)
endif (BUILD_LIBREALSENSE)

if (BUILD_AZURE_KINECT)
    EXAMPLE_CPP(AzureKinectRecord     ${CMAKE_PROJECT_NAME})
    EXAMPLE_CPP(AzureKinectMKVReader  ${CMAKE_PROJECT_NAME})
    EXAMPLE_CPP(AzureKinectViewer     ${CMAKE_PROJECT_NAME})
endif (BUILD_AZURE_KINECT)

EXAMPLE_CPP(RegistrationRANSAC        ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(RGBDOdometry              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(TriangleMesh              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(TrimMeshBasedOnPointCloud ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ViewDistances             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ViewPCDMatch              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Visualizer                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Voxelization              ${CMAKE_PROJECT_NAME})

if (ENABLE_GUI)
    if (WIN32)
        EXAMPLE_CPP(FilamentDemo      ${CMAKE_PROJECT_NAME} SDL2main$<$<CONFIG:Debug>:d>)
    else()
        EXAMPLE_CPP(FilamentDemo      ${CMAKE_PROJECT_NAME})
    endif()
    EXAMPLE_CPP(GuiDemo               ${CMAKE_PROJECT_NAME})

    set(GUIDEMO_RSRC_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/resources)
    if (WIN32)
        add_custom_target(GuiDemoResources
                          ${CMAKE_COMMAND} -E copy_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources ${GUIDEMO_RSRC_DIR})
    else ()
        add_custom_target(GuiDemoResources
                          rm -f ${GUIDEMO_RSRC_DIR}
                          COMMAND ln -s ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources ${GUIDEMO_RSRC_DIR})
    endif()
    add_dependencies(GuiDemo GuiDemoResources)

    # Copy .dll files on Windows
    if (WIN32)
        add_custom_target(copy_dll
            COMMAND xcopy /s /i /y /f \"bin/*.dll\" \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/${CMAKE_BUILD_TYPE}\"
            WORKING_DIRECTORY ${3RDPARTY_INSTALL_PREFIX}
            DEPENDS ext_sdl2)

        add_dependencies(GuiDemo copy_dll)
    endif()
endif (ENABLE_GUI)
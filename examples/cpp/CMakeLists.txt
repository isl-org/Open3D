include(ProcessorCount)
ProcessorCount(NPROC)

set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")

macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")
    open3d_show_and_abort_on_warning(${EXAMPLE_CPP_NAME})
    open3d_set_global_properties(${EXAMPLE_CPP_NAME})

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
        FOLDER "examples/cpp/"
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BIN_DIR}"
    )

    if(NOT BUILD_EXAMPLES)
        set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
    endif()

    list(APPEND EXAMPLE_TARGETS ${EXAMPLE_CPP_NAME})
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(CameraPoseTrajectory      ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ColorMapOptimization      ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(DepthCapture              ${CMAKE_PROJECT_NAME})
if (BUILD_GUI)
    EXAMPLE_CPP(Draw                  ${CMAKE_PROJECT_NAME})
endif()
EXAMPLE_CPP(TrimMeshBasedOnPointCloud ${CMAKE_PROJECT_NAME} ${OPENMP_TARGET})
EXAMPLE_CPP(EvaluatePCDMatch          ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(FileDialog                ${CMAKE_PROJECT_NAME} ${TINYFILEDIALOGS_TARGET})
EXAMPLE_CPP(FileSystem                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Flann                     ${CMAKE_PROJECT_NAME} ${FLANN_TARGET})
EXAMPLE_CPP(Image                     ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(IntegrateRGBD             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ISSKeypoints              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(LineSet                   ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Log                       ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(OdometryRGBD              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Octree                    ${CMAKE_PROJECT_NAME})
if (BUILD_GUI)
    EXAMPLE_CPP(OffscreenRendering    ${CMAKE_PROJECT_NAME})
endif()
EXAMPLE_CPP(PCDFileFormat             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(PointCloud                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(PoseGraph                 ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ProgramOptions            ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(RegistrationRANSAC        ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(RGBDOdometry              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(TriangleMesh              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(TIntegrateRGBD            ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(EvaluateFeatureMatch      ${CMAKE_PROJECT_NAME} ${OPENMP_TARGET} ${FLANN_TARGET})
EXAMPLE_CPP(ViewDistances             ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(ViewPCDMatch              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Visualizer                ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(Voxelization              ${CMAKE_PROJECT_NAME})
EXAMPLE_CPP(OpenMP                    ${CMAKE_PROJECT_NAME} ${OPENMP_TARGET})

if(BUILD_LIBREALSENSE)
    EXAMPLE_CPP(RealSenseRecorder     ${CMAKE_PROJECT_NAME} ${LIBREALSENSE_TARGET})
    EXAMPLE_CPP(RealSenseBagReader    ${CMAKE_PROJECT_NAME} ${LIBREALSENSE_TARGET} ${JSONCPP_TARGET} ${OPENMP_TARGET})
endif()
if(BUILD_AZURE_KINECT)
    EXAMPLE_CPP(AzureKinectRecord     ${CMAKE_PROJECT_NAME} ${K4A_TARGET})
    EXAMPLE_CPP(AzureKinectMKVReader  ${CMAKE_PROJECT_NAME} ${JSONCPP_TARGET})
    EXAMPLE_CPP(AzureKinectViewer     ${CMAKE_PROJECT_NAME} ${K4A_TARGET})
endif()

# build-examples-iteratively is used to conserve space on CI machine.
add_custom_target(build-examples-iteratively
    COMMAND ${CMAKE_COMMAND}
    -DEXAMPLE_TARGETS="${EXAMPLE_TARGETS}"
    -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    -DEXAMPLE_BIN_DIR="${EXAMPLE_BIN_DIR}"
    -DCMAKE_BUILD_TYPE="$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>$<$<CONFIG:MinSizeRel>:MinSizeRel>"
    -DNPROC="${NPROC}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/iterative_build_examples.cmake
)

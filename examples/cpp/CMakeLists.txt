set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")

macro(open3d_add_example EXAMPLE_CPP_NAME)
    add_executable(${EXAMPLE_CPP_NAME})

    target_sources(${EXAMPLE_CPP_NAME} PRIVATE "${EXAMPLE_CPP_NAME}.cpp")
    target_link_libraries(${EXAMPLE_CPP_NAME} PRIVATE Open3D::Open3D ${ARGN})

    # Don't use open3d_link_3rdparty_libraries(${EXAMPLE_CPP_NAME}).
    # Unlike benchmarks, examples are intended as standalone apps that link
    # the main Open3D library, while the benchmarks executable needs 3rd-party
    # libraries since it measures the internal componets of Open3D.
    open3d_show_and_abort_on_warning(${EXAMPLE_CPP_NAME})
    open3d_set_global_properties(${EXAMPLE_CPP_NAME})

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
        FOLDER "examples/cpp/"
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BIN_DIR}"
    )

    if (NOT BUILD_EXAMPLES)
        set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
    endif()

    list(APPEND EXAMPLE_TARGETS ${EXAMPLE_CPP_NAME})
endmacro()

open3d_add_example(CameraPoseTrajectory)
open3d_add_example(ColorMapOptimization)
open3d_add_example(DepthCapture)
open3d_add_example(EvaluateFeatureMatch      ${OPENMP_TARGET} ${FLANN_TARGET})
open3d_add_example(EvaluatePCDMatch)
open3d_add_example(FileDialog                ${TINYFILEDIALOGS_TARGET})
open3d_add_example(FileSystem)
open3d_add_example(Flann                     ${FLANN_TARGET})
open3d_add_example(Image)
open3d_add_example(IntegrateRGBD)
open3d_add_example(ISSKeypoints)
open3d_add_example(LineSet)
open3d_add_example(Log)
open3d_add_example(Octree)
open3d_add_example(OdometryRGBD)
open3d_add_example(OpenMP                    ${OPENMP_TARGET})
open3d_add_example(PCDFileFormat)
open3d_add_example(PointCloud)
open3d_add_example(PoseGraph)
open3d_add_example(ProgramOptions)
open3d_add_example(RegistrationColoredICP)
open3d_add_example(RegistrationRANSAC)
open3d_add_example(RGBDOdometry)
open3d_add_example(SLAC)
open3d_add_example(SLACIntegrate)
open3d_add_example(TIntegrateRGBD)
open3d_add_example(TOdometryRGBD)
open3d_add_example(TriangleMesh)
open3d_add_example(TrimMeshBasedOnPointCloud ${OPENMP_TARGET})
open3d_add_example(ViewDistances)
open3d_add_example(ViewPCDMatch)
open3d_add_example(Visualizer)
open3d_add_example(VoxelHashing)
open3d_add_example(Voxelization)

if (BUILD_GUI)
    open3d_add_example(Draw)
    open3d_add_example(MultipleWindows       Threads::Threads)
    open3d_add_example(OffscreenRendering)
    open3d_add_example(TICPOdometry          Threads::Threads)
    open3d_add_example(TICPReconstruction    Threads::Threads)
endif()

if(BUILD_WEBRTC)
    open3d_add_example(DrawWebRTC)
endif()

if (BUILD_LIBREALSENSE)
    open3d_add_example(RealSenseBagReader    ${LIBREALSENSE_TARGET} ${JSONCPP_TARGET} ${OPENMP_TARGET})
    open3d_add_example(RealSenseRecorder     ${LIBREALSENSE_TARGET})
endif()

if (BUILD_AZURE_KINECT)
    open3d_add_example(AzureKinectMKVReader  ${JSONCPP_TARGET})
    open3d_add_example(AzureKinectRecord     ${K4A_TARGET})
    open3d_add_example(AzureKinectViewer     ${K4A_TARGET})
endif()

include(ProcessorCount)
ProcessorCount(NPROC)

# build-examples-iteratively is used to conserve space on CI machine.
add_custom_target(build-examples-iteratively
    COMMAND ${CMAKE_COMMAND}
    -DEXAMPLE_TARGETS="${EXAMPLE_TARGETS}"
    -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    -DEXAMPLE_BIN_DIR="${EXAMPLE_BIN_DIR}"
    -DCMAKE_BUILD_TYPE="$<CONFIG>"
    -DNPROC="${NPROC}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/iterative_build_examples.cmake
)

# Internal helper function.
function(open3d_aligned_print printed_name printed_valued)
    string(LENGTH "${printed_name}" PRINTED_NAME_LENGTH)
    math(EXPR PRINTED_DOTS_LENGTH "40 - ${PRINTED_NAME_LENGTH}")
    string(REPEAT "." ${PRINTED_DOTS_LENGTH} PRINTED_DOTS)
    message(STATUS "  ${printed_name} ${PRINTED_DOTS} ${printed_valued}")
endfunction()


# open3d_print_configuration_summary()
#
# Prints a summary of the current configuration.
function(open3d_print_configuration_summary)
    message(STATUS "================================================================================")
    message(STATUS "Open3D ${PROJECT_VERSION} Configuration Summary")
    message(STATUS "================================================================================")
    message(STATUS "Enabled Features:")
    open3d_aligned_print("OpenMP" "${WITH_OPENMP}")
    open3d_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
    open3d_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
    open3d_aligned_print("Intel RealSense Support" "${BUILD_LIBREALSENSE}")
    open3d_aligned_print("CUDA Support" "${BUILD_CUDA_MODULE}")
    if(BUILD_CUDA_MODULE)
        open3d_aligned_print("CUDA cached memory manager" "${ENABLE_CACHED_CUDA_MANAGER}")
    endif()
    open3d_aligned_print("SYCL Support" "${BUILD_SYCL_MODULE}")
    if(ENABLE_SYCL_UNIFIED_SHARED_MEMORY)
        open3d_aligned_print("SYCL unified shared memory" "${ENABLE_SYCL_UNIFIED_SHARED_MEMORY}")
    endif()
    open3d_aligned_print("ISPC Support" "${BUILD_ISPC_MODULE}")
    open3d_aligned_print("Build GUI" "${BUILD_GUI}")
    open3d_aligned_print("Build WebRTC visualizer" "${BUILD_WEBRTC}")
    open3d_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
    if(WIN32)
       open3d_aligned_print("Use Windows Static Runtime" "${STATIC_WINDOWS_RUNTIME}")
    endif()
    open3d_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
    open3d_aligned_print("Build Examples" "${BUILD_EXAMPLES}")
    open3d_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
    open3d_aligned_print("Build Jupyter Extension" "${BUILD_JUPYTER_EXTENSION}")
    open3d_aligned_print("Build TensorFlow Ops" "${BUILD_TENSORFLOW_OPS}")
    open3d_aligned_print("Build PyTorch Ops" "${BUILD_PYTORCH_OPS}")
    open3d_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
    open3d_aligned_print("Bundle Open3D-ML" "${BUNDLE_OPEN3D_ML}")
    if(GLIBCXX_USE_CXX11_ABI)
        open3d_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "1")
    else()
        open3d_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "0")
    endif()

    message(STATUS "================================================================================")
    message(STATUS "Third-Party Dependencies:")
    set(3RDPARTY_DEPENDENCIES
        Assimp
        BLAS
        curl
        Eigen3
        filament
        fmt
        GLEW
        GLFW
        googletest
        imgui
        ipp
        JPEG
        jsoncpp
        liblzf
        msgpack
        nanoflann
        OpenGL
        PNG
        qhullcpp
        librealsense
        TBB
        tinyfiledialogs
        TinyGLTF
        tinyobjloader
        VTK
        WebRTC
        ZeroMQ
    )
    foreach(dep IN LISTS 3RDPARTY_DEPENDENCIES)
        string(TOLOWER "${dep}" dep_lower)
        string(TOUPPER "${dep}" dep_upper)
        if(TARGET Open3D::3rdparty_${dep_lower})
            if(NOT USE_SYSTEM_${dep_upper})
                open3d_aligned_print("${dep}" "yes (build from source)")
            else()
                if(3rdparty_${dep_lower}_VERSION)
                    open3d_aligned_print("${dep}" "yes (v${3rdparty_${dep_lower}_VERSION})")
                else()
                    open3d_aligned_print("${dep}" "yes")
                endif()
            endif()
        else()
            open3d_aligned_print("${dep}" "no")
        endif()
    endforeach()
    message(STATUS "================================================================================")

endfunction()

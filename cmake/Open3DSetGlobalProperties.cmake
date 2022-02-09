# open3d_set_global_properties(target)
#
# Sets important project-related properties to <target>.
function(open3d_set_global_properties target)
    # Tell CMake we want a compiler that supports C++14 features
    target_compile_features(${target} PUBLIC cxx_std_14)

    # Detect compiler id and version for utility::CompilerInfo
    # - OPEN3D_CXX_STANDARD
    # - OPEN3D_CXX_COMPILER_ID
    # - OPEN3D_CXX_COMPILER_VERSION
    # - OPEN3D_CUDA_COMPILER_ID       # Emtpy if not BUILD_CUDA_MODULE
    # - OPEN3D_CUDA_COMPILER_VERSION  # Emtpy if not BUILD_CUDA_MODULE
    if (NOT CMAKE_CXX_STANDARD)
        message(FATAL_ERROR "CMAKE_CXX_STANDARD must be defined globally.")
    endif()
    target_compile_definitions(${target} PRIVATE OPEN3D_CXX_STANDARD="${CMAKE_CXX_STANDARD}")
    target_compile_definitions(${target} PRIVATE OPEN3D_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
    target_compile_definitions(${target} PRIVATE OPEN3D_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
    target_compile_definitions(${target} PRIVATE OPEN3D_CUDA_COMPILER_ID="${CMAKE_CUDA_COMPILER_ID}")
    target_compile_definitions(${target} PRIVATE OPEN3D_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}")

    # std::filesystem (C++17) or std::experimental::filesystem (C++14)
    #
    # Ref: https://en.cppreference.com/w/cpp/filesystem:
    #      Using this library may require additional compiler/linker options.
    #      GNU implementation prior to 9.1 requires linking with -lstdc++fs and
    #      LLVM implementation prior to LLVM 9.0 requires linking with -lc++fs.
    # Ref: https://gitlab.kitware.com/cmake/cmake/-/issues/17834
    #      It's non-trivial to determine the link flags for CMake.
    #
    # The linkage can be "-lstdc++fs" or "-lc++fs" or ""(empty). In our
    # experiments, the behaviour doesn't quite match the specifications.
    #
    # - On Ubuntu 20.04:
    #   - "-lstdc++fs" works with with GCC 7/10 and Clang 7/12
    #   - "" does not work with GCC 7/10 and Clang 7/12
    #
    # - On latest macOS/Windows with the default compiler:
    #   - "" works.
    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PRIVATE stdc++fs)
    endif()

    # Colorize GCC/Clang terminal outputs
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
    endif()

    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )

    # Required for static linking zeromq
    target_compile_definitions(${target} PRIVATE ZMQ_STATIC)

    # Propagate build configuration into source code
    if (BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
        if (BUILD_CACHED_CUDA_MANAGER)
            target_compile_definitions(${target} PRIVATE BUILD_CACHED_CUDA_MANAGER)
        endif()
    endif()
    if (BUILD_ISPC_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_ISPC_MODULE)
    endif()
    if (BUILD_GUI)
        target_compile_definitions(${target} PRIVATE BUILD_GUI)
    endif()
    if (ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif()
    if (BUILD_AZURE_KINECT)
        target_compile_definitions(${target} PRIVATE BUILD_AZURE_KINECT)
    endif()
    if (BUILD_LIBREALSENSE)
        target_compile_definitions(${target} PRIVATE BUILD_LIBREALSENSE)
    endif()
    if (BUILD_WEBRTC)
        target_compile_definitions(${target} PRIVATE BUILD_WEBRTC)
    endif()
    if (USE_BLAS)
        target_compile_definitions(${target} PRIVATE USE_BLAS)
    endif()
    if (WITH_IPPICV)
        target_compile_definitions(${target} PRIVATE WITH_IPPICV)
    endif()
    if (GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()

    if(NOT WITH_OPENMP)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>")
    endif()
    if(WIN32)
        target_compile_definitions(${target} PRIVATE
            WINDOWS
            _CRT_SECURE_NO_DEPRECATE
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
        )
        if(MSVC)
            target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # Option 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
            if(BUILD_GUI)
                # GLEW and Open3D make direct OpenGL calls and link to opengl32.lib;
                # Filament needs to link through bluegl.lib.
                # See https://github.com/google/filament/issues/652
                target_link_options(${target} PRIVATE /force:multiple)
            endif()
            # The examples' .pdb files use up a lot of space and cause us to run
            # out of space on Github Actions. Compressing gives us another couple of GB.
            target_link_options(${target} PRIVATE /pdbcompress)
        endif()
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
    elseif(UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
    endif()
    if(LINUX_AARCH64)
        target_compile_definitions(${target} PRIVATE LINUX_AARCH64)
    endif()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")

    # Require 64-bit indexing in vectorized code
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--addressing=64>)

    # Set architecture flag
    if(LINUX_AARCH64)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--arch=aarch64>)
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--arch=x86-64>)
    endif()

    # TBB static version is used
    # See: https://github.com/wjakob/tbb/commit/615d690c165d68088c32b6756c430261b309b79c
    target_compile_definitions(${target} PRIVATE __TBB_LIB_NAME=tbb_static)

    # Download test data files from open3d_downloads repo.
    add_dependencies(${target} open3d_downloads)

    # Strip unnecessary sections of the binary on Linux/macOS for Release builds
    # (from pybind11)
    # macOS: -x: strip local symbols
    # Linux: defaults
    if(NOT DEVELOPER_BUILD AND UNIX AND CMAKE_STRIP)
        get_target_property(target_type ${target} TYPE)
        if(target_type MATCHES
                MODULE_LIBRARY|SHARED_LIBRARY|EXECUTABLE)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND
                $<IF:$<CONFIG:Release>,${CMAKE_STRIP},true>
                $<$<PLATFORM_ID:Darwin>:-x> $<TARGET_FILE:${target}>
                COMMAND_EXPAND_LISTS)
        endif()
    endif()

endfunction()

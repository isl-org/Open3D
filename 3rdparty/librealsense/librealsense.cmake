include(ExternalProject)

find_package(Git QUIET REQUIRED)

# Find libusb-1.0 for USE_EXTERNAL_USB=ON
# On Ubuntu/Debian, libusb-1.0 headers are typically in /usr/include/libusb-1.0/
# and the header file is libusb-1.0/libusb.h, but code expects to find <libusb.h>
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBUSB1 QUIET libusb-1.0)
endif()

# Fallback: try to find libusb headers manually
if(NOT LIBUSB1_FOUND)
    # First check for libusb-1.0 subdirectory (Ubuntu/Debian standard location)
    if(EXISTS "/usr/include/libusb-1.0/libusb.h")
        set(LIBUSB1_INCLUDE_DIRS "/usr/include/libusb-1.0")
        set(LIBUSB1_FOUND TRUE)
    # Then check for direct libusb.h
    else()
        find_path(LIBUSB1_INCLUDE_DIR
            NAMES libusb.h
            PATHS
                /usr/include
                /usr/local/include
        )
        if(LIBUSB1_INCLUDE_DIR)
            set(LIBUSB1_INCLUDE_DIRS ${LIBUSB1_INCLUDE_DIR})
            set(LIBUSB1_FOUND TRUE)
        endif()
    endif()
endif()

# Set libusb include flags if found
# For Linux/Ubuntu Docker builds, libusb-1.0 headers are in /usr/include/libusb-1.0/
set(LIBUSB1_CMAKE_C_FLAGS_EXTRA "")
set(LIBUSB1_CMAKE_CXX_FLAGS_EXTRA "")
if(LIBUSB1_FOUND AND LIBUSB1_INCLUDE_DIRS)
    # Add include directory to C/C++ flags as a string
    set(LIBUSB1_CMAKE_C_FLAGS_EXTRA "-I${LIBUSB1_INCLUDE_DIRS}")
    set(LIBUSB1_CMAKE_CXX_FLAGS_EXTRA "-I${LIBUSB1_INCLUDE_DIRS}")
endif()

# Construct CMAKE_C_FLAGS and CMAKE_CXX_FLAGS with libusb includes for non-Windows
set(LIBREALSENSE_CMAKE_C_FLAGS "")
set(LIBREALSENSE_CMAKE_CXX_FLAGS "")
set(LIBREALSENSE_EXTRA_CMAKE_ARGS "")
if(NOT WIN32)
    if(LIBUSB1_CMAKE_C_FLAGS_EXTRA)
        set(LIBREALSENSE_CMAKE_C_FLAGS "${LIBUSB1_CMAKE_C_FLAGS_EXTRA}")
        list(APPEND LIBREALSENSE_EXTRA_CMAKE_ARGS "-DCMAKE_C_FLAGS=${LIBUSB1_CMAKE_C_FLAGS_EXTRA}")
    endif()
    # CXX flags include both GLIBCXX and libusb
    set(LIBREALSENSE_CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>")
    if(LIBUSB1_CMAKE_CXX_FLAGS_EXTRA)
        set(LIBREALSENSE_CMAKE_CXX_FLAGS "${LIBREALSENSE_CMAKE_CXX_FLAGS} ${LIBUSB1_CMAKE_CXX_FLAGS_EXTRA}")
    endif()
endif()

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    URL https://github.com/realsenseai/librealsense/archive/refs/tags/v2.57.4.tar.gz #  2023 Sep 28
    # Future versions after v2.54.2 may not support L515 and SR300
    URL_HASH SHA256=3e82f9b545d9345fd544bb65f8bf7943969fb40bcfc73d983e7c2ffcdc05eaeb
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/librealsense"
    UPDATE_COMMAND ""
    # Patch for CRT mismatch in CUDA code (Windows)
    COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> init
    COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_LIST_DIR}/fix-cudacrt.patch
    # Patch to include the <chrono> header for the system_clock type
    COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_LIST_DIR}/fix-include-chrono.patch
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_GLSL_EXTENSIONS=OFF
        -DBUILD_GRAPHICAL_EXAMPLES=OFF
        -DBUILD_PYTHON_BINDINGS=OFF
        -DBUILD_WITH_CUDA=${BUILD_CUDA_MODULE}
        -DUSE_EXTERNAL_USB=ON
        -DBUILD_TOOLS=OFF
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
        # checking CXX_COMPILER_ID is not supported.
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=${LIBREALSENSE_CMAKE_CXX_FLAGS}>
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
        ${LIBREALSENSE_EXTRA_CMAKE_ARGS}
        ${ExternalProject_CMAKE_ARGS_hidden}
    CMAKE_CACHE_ARGS    # Lists must be passed via CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}realsense2${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}realsense-file${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}fw${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rsutils${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_librealsense INSTALL_DIR)
set(LIBREALSENSE_INCLUDE_DIR "${INSTALL_DIR}/include/") # "/" is critical.
set(LIBREALSENSE_LIB_DIR "${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR}")
set(LIBREALSENSE_LIBRARIES realsense2 fw realsense-file rsutils) # The order is critical.
# Note: usb-1.0 is a system library and should be linked separately, not included in LIBRARIES
# which expects static library files in LIB_DIR
if(MSVC)    # Rename debug libs to ${LIBREALSENSE_LIBRARIES}. rem (comment) is no-op
    ExternalProject_Add_Step(ext_librealsense rename_debug_libs
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense2d.lib realsense2.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y fwd.lib fw.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense-filed.lib realsense-file.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y rsutilsd.lib rsutils.lib
        WORKING_DIRECTORY "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
endif()

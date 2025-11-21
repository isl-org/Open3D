include(ExternalProject)

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    URL https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.54.2.tar.gz #  2023 Sep 28
    # Future versions after v2.54.2 may not support L515 and SR300
    URL_HASH SHA256=e3a767337ff40ae41000049a490ab84bd70b00cbfef65e8cdbadf17fd2e1e5a8
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/librealsense"
    UPDATE_COMMAND ""
    # Patch for libusb static build failure on Linux
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librealsense/libusb-CMakeLists.txt
        <SOURCE_DIR>/third-party/libusb/CMakeLists.txt
    # Patch for CRT mismatch in CUDA code (Windows)
    COMMAND ${GIT_EXECUTABLE} init
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/fix-cudacrt.patch
        ${CMAKE_CURRENT_LIST_DIR}/fix_mac_apple_silicon_build.patch
        # Patch to include the <chrono> header for the system_clock type
        ${CMAKE_CURRENT_LIST_DIR}/fix-include-chrono.patch
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
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>>
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
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
if (NOT MSVC)
	list(APPEND LIBREALSENSE_LIBRARIES usb)
endif()
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

if (NOT MSVC)
	ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
	    COMMAND ${CMAKE_COMMAND} -E copy
	    "<BINARY_DIR>/libusb_install/lib/${CMAKE_STATIC_LIBRARY_PREFIX}usb${CMAKE_STATIC_LIBRARY_SUFFIX}"
	    "${LIBREALSENSE_LIB_DIR}"
	    DEPENDEES install
	    BYPRODUCTS "${LIBREALSENSE_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}usb${CMAKE_STATIC_LIBRARY_SUFFIX}"
	    )
endif()

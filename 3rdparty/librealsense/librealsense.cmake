include(ExternalProject)

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    GIT_REPOSITORY https://github.com/IntelRealSense/librealsense.git
    GIT_TAG v2.40.0 # 18 Nov 2020
    UPDATE_COMMAND ""
    # Patch for libusb static build failure on Linux
    PATCH_COMMAND git -C <SOURCE_DIR> reset --hard v2.40.0
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librealsense/libusb-CMakeLists.txt
    <SOURCE_DIR>/third-party/libusb/CMakeLists.txt
    # Patch for libstdc++ regex bug
    COMMAND git -C <SOURCE_DIR> apply
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librealsense/fix-2837.patch
    CMAKE_ARGS
        -D CMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -D BUILD_SHARED_LIBS=OFF
        -D BUILD_EXAMPLES=OFF
        -D BUILD_UNIT_TESTS=OFF
        -D BUILD_GLSL_EXTENSIONS=OFF
        -D BUILD_GRAPHICAL_EXAMPLES=OFF
        -D BUILD_PYTHON_BINDINGS=OFF
        -D BUILD_WITH_CUDA=${BUILD_CUDA_MODULE}
        -D FORCE_RSUSB_BACKEND=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>      # https://github.com/IntelRealSense/librealsense/wiki/Release-Notes#release-2400
        -D USE_EXTERNAL_USB=ON
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
)

ExternalProject_Get_Property(ext_librealsense INSTALL_DIR)
set(LIBREALSENSE_INCLUDE_DIR "${INSTALL_DIR}/include/") # "/" is critical.
set(LIBREALSENSE_LIB_DIR "${INSTALL_DIR}/lib")

set(LIBREALSENSE_LIBRARIES realsense2 fw realsense-file usb) # The order is critical.
if(MSVC)    # Rename debug libs to ${LIBREALSENSE_LIBRARIES}. rem (comment) is no-op
    ExternalProject_Add_Step(ext_librealsense rename_debug_libs
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense2d.lib realsense2.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y fwd.lib fw.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense-filed.lib realsense-file.lib
        WORKING_DIRECTORY "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
endif()

ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
    COMMAND ${CMAKE_COMMAND} -E copy
    "<BINARY_DIR>/libusb_install/lib/${CMAKE_STATIC_LIBRARY_PREFIX}usb${CMAKE_STATIC_LIBRARY_SUFFIX}"
    "${LIBREALSENSE_LIB_DIR}"
    DEPENDEES install
    )

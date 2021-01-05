include(ExternalProject)

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    GIT_REPOSITORY https://github.com/IntelRealSense/librealsense.git
    GIT_TAG v2.40.0 # 18 Nov 2020
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librealsense/libusb-CMakeLists.txt
        <SOURCE_DIR>/third-party/libusb/CMakeLists.txt
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_GLSL_EXTENSIONS=OFF
        -DBUILD_GRAPHICAL_EXAMPLES=OFF
        -DBUILD_PYTHON_BINDINGS=ON
        -DBUILD_WITH_CUDA=${BUILD_CUDA_MODULE}
        -DPYBIND11_PYTHON_VERSION=3.6
        -DFORCE_RSUSB_BACKEND=ON      # https://github.com/IntelRealSense/librealsense/wiki/Release-Notes#release-2400
        -DUSE_EXTERNAL_USB=ON
        $<$<PLATFORM_ID:Darwin>:-DUSE_EXTERNAL_USB=ON>
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DUSE_EXTERNAL_USB=ON>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
)

ExternalProject_Get_Property(ext_librealsense INSTALL_DIR)
set(LIBREALSENSE_INCLUDE_DIR "${INSTALL_DIR}/include/") # "/" is critical.
set(LIBREALSENSE_LIB_DIR "${INSTALL_DIR}/lib")

set(LIBREALSENSE_LIBRARIES realsense2 fw realsense-file) # The order is critical.
if(MSVC)    # Rename debug libs to ${LIBREALSENSE_LIBRARIES}. rem (comment) is no-op
    ExternalProject_Add_Step(ext_librealsense rename_debug_libs
        COMMAND $<IF:$<CONFIG:Debug>,rename,rem> realsense2d.lib realsense2.lib
        COMMAND $<IF:$<CONFIG:Debug>,rename,rem> fwd.lib fw.lib
        COMMAND $<IF:$<CONFIG:Debug>,rename,rem> realsense-filed.lib realsense-file.lib
        WORKING_DIRECTORY "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
endif()

if(WIN32)
    ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
        COMMAND ${CMAKE_COMMAND} -E copy
        "<BINARY_DIR>/libusb_install/lib/usb.lib" "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
    set(LIBREALSENSE_LIBRARIES ${LIBREALSENSE_LIBRARIES} usb)
else() # if(APPLE) or Linux
    ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
        COMMAND ${CMAKE_COMMAND} -E copy
        "<BINARY_DIR>/libusb_install/lib/libusb.a" "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
    set(LIBREALSENSE_LIBRARIES ${LIBREALSENSE_LIBRARIES} usb)
endif()

include(ExternalProject)

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    GIT_REPOSITORY https://github.com/IntelRealSense/librealsense.git
    GIT_TAG v2.39.0 # Oct 2020
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_GLSL_EXTENSIONS=OFF
        -DBUILD_GRAPHICAL_EXAMPLES=OFF
        -DBUILD_PYTHON_BINDINGS=OFF
        -DBUILD_WITH_CUDA=${BUILD_CUDA_MODULE}
        $<$<PLATFORM_ID:Darwin>:-DUSE_EXTERNAL_USB=ON>
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DUSE_EXTERNAL_USB=ON>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
)

ExternalProject_Get_Property(ext_librealsense INSTALL_DIR)
set(LIBREALSENSE_INCLUDE_DIR ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBREALSENSE_LIB_DIR ${INSTALL_DIR}/lib)
set(LIBREALSENSE_LIBRARIES realsense2 fw realsense-file) # The order is critical.

if(APPLE)
    ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
        COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libusb_install/lib/libusb.a ${LIBREALSENSE_LIB_DIR}
        DEPENDEES install
    )
    set(LIBREALSENSE_LIBRARIES ${LIBREALSENSE_LIBRARIES} usb)
elseif(WIN32)
    ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
        COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libusb_install/lib/usb.lib ${LIBREALSENSE_LIB_DIR}
        DEPENDEES install
    )
    set(LIBREALSENSE_LIBRARIES ${LIBREALSENSE_LIBRARIES} usb)
endif()

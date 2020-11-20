# Azure Kinect 3rd-party library support.
#
# Azure Kinect source code are used only for header include, which is sufficient
# for buliding Open3D with Kinect support.
#
# To run Open3D with Kinect support, users have to download and install Azure
# Kinect SDK (k4a) . Open3D dlopens k4a library at runtime.
#
# This CMake script exports the following variable(s):
# - K4A_INCLUDE_DIR
include(ExternalProject)

if (APPLE)
    message(WARNING "Azure Kinect is not supported on macOS, setting BUILD_AZURE_KINECT to OFF")
    set(BUILD_AZURE_KINECT OFF)
    set(BUILD_AZURE_KINECT OFF PARENT_SCOPE)
endif()

# Conditionally include header files in Open3D.h, when azure kinect is enabled.
set(BUILD_AZURE_KINECT_COMMENT "")

# Export the following variables:
# - k4a_INCLUDE_DIRS
if (WIN32)
    # This works even when the user does not have k4a libraries installed
    # in `Program Files`. We only need the headers.
    ExternalProject_Add(
        ext_k4a
        PREFIX k4a
        URL https://www.nuget.org/api/v2/package/Microsoft.Azure.Kinect.Sensor/1.4.1
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Get_Property(ext_k4a SOURCE_DIR)
    set(K4A_INCLUDE_DIR ${SOURCE_DIR}/build/native/include/)
else()
    # Try to find system-wide installed K4a.
    # The property names are tested with k4a 1.4.1, future versions might work.
    find_package(k4a QUIET)
    find_package(k4arecord QUIET)
    if (k4a_FOUND)
        get_target_property(k4a_INCLUDE_DIRS k4a::k4a INTERFACE_INCLUDE_DIRECTORIES)
    endif()

    if (k4a_FOUND)
        message(STATUS "k4a_INCLUDE_DIRS: ${k4a_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "Kinect SDK NOT found. Please install according \
                to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md")
    endif()
endif()

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

# This works even when the user does not have k4a libraries installed
# in `Program Files`. We only need the headers.
if (WIN32)
    ExternalProject_Add(
        ext_k4a
        PREFIX k4a
        URL https://www.nuget.org/api/v2/package/Microsoft.Azure.Kinect.Sensor/1.4.1
        URL_HASH SHA256=6c512a20c4a82b80e02b0f6d4a6cda51e88d4893cd47ab85c7bca37cd364c976
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/k4a"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Get_Property(ext_k4a SOURCE_DIR)
    set(K4A_INCLUDE_DIR ${SOURCE_DIR}/build/native/include/) # "/" is critical
else()
    ExternalProject_Add(
        ext_k4a
        PREFIX k4a
        URL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
        URL_HASH SHA256=08303094b9ad36ea74c19bc8b8950c97055e73dd2e8bd18e2af5e165a2289cd2
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/k4a"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Add_Step(ext_k4a extract_headers
        COMMAND ${CMAKE_COMMAND} -E tar xvf data.tar.gz
        WORKING_DIRECTORY <SOURCE_DIR>
        DEPENDEES download
        DEPENDERS update
        INDEPENDENT TRUE
    )
    ExternalProject_Get_Property(ext_k4a SOURCE_DIR)
    set(K4A_INCLUDE_DIR ${SOURCE_DIR}/usr/include/) # "/" is critical
endif()

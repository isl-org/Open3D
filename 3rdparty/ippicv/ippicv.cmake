# Adapted from: https://github.com/opencv/opencv/blob/master/3rdparty/ippicv/ippicv.cmake
# Downloads IPPICV libraries from the OpenCV 3rd party repo

include(ExternalProject)

# Commit SHA in the opencv_3rdparty repo
set(IPPICV_COMMIT "a56b6ac6f030c312b2dce17430eef13aed9af274")
# Check in order APPLE -> WIN32 -> UNIX, since UNIX may be defined on APPLE /
# WIN32 as well
if(APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(OPENCV_ICV_NAME "ippicv_2020_mac_intel64_20191018_general.tgz")
    set(OPENCV_ICV_HASH "3a39aad1eef2f6019dda9555f6db2d34063c1e464dc0e498eaa0c6b55817f2fe")
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL AMD64)
    set(OPENCV_ICV_NAME "ippicv_2020_win_intel64_20191018_general.zip")
    set(OPENCV_ICV_HASH "e64e09f8a2e121d4fff440fb12b1298bc0760f1391770aefe5d1deb6630352b7")
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES 86) # {x86, i386, i686}
    set(OPENCV_ICV_NAME "ippicv_2020_win_ia32_20191018_general.zip")
    set(OPENCV_ICV_HASH "c1e0e26f32aec4374df05a145cfac09774e15f9b53f0bdfaac3eca3205db6106")
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(OPENCV_ICV_NAME "ippicv_2020_lnx_intel64_20191018_general.tgz")
    set(OPENCV_ICV_HASH "08627fa5660d52d59309a572dd7db5b9c8aea234cfa5aee0942a1dd903554246")
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES 86) # {x86, i386, i686}
    set(OPENCV_ICV_NAME "ippicv_2020_lnx_ia32_20191018_general.tgz")
    set(OPENCV_ICV_HASH "acf8976ddea689b6d8c9640d6cfa6852d5c74c1fc368b57029b13ed7714fbd95")
else()
    set(WITH_IPPICV OFF)
    message(FATAL_ERROR "IPP-ICV disabled: Unsupported Platform.")
    return()
endif()

if(WIN32)
    set(lib_name ippicvmt)
else()
    set(lib_name ippicv)
endif()

ExternalProject_Add(ext_ippicv
    PREFIX ippicv
    URL https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/${OPENCV_ICV_NAME}
    URL_HASH SHA256=${OPENCV_ICV_HASH}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/ippicv"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ippicv/CMakeLists.txt <SOURCE_DIR>
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ippiw${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

ExternalProject_Get_Property(ext_ippicv INSTALL_DIR)
set(IPPICV_INCLUDE_DIR "${INSTALL_DIR}/include/icv/" "${INSTALL_DIR}/include/")
set(IPPICV_LIBRARIES ippiw ${lib_name})
set(IPPICV_LIB_DIR "${INSTALL_DIR}/lib")
set(IPPICV_VERSION_STRING "2020.0.0 Gold")  # From icv/ippversion.h

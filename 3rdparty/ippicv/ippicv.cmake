# Adapted from: https://github.com/opencv/opencv/blob/master/3rdparty/ippicv/ippicv.cmake
# Downloads IPPICV libraries from the OpenCV 3rd party repo

include(ExternalProject)
# Commit SHA in the opencv_3rdparty repo
set(IPPICV_COMMIT "a56b6ac6f030c312b2dce17430eef13aed9af274")
# Check in order APPLE -> WIN32 -> UNIX, since UNIX may be defined on APPLE /
# WIN32 as well
if(APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(OPENCV_ICV_NAME "ippicv_2020_mac_intel64_20191018_general.tgz")
    set(OPENCV_ICV_HASH "1c3d675c2a2395d094d523024896e01b")
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL AMD64)
    set(OPENCV_ICV_NAME "ippicv_2020_win_intel64_20191018_general.zip")
    set(OPENCV_ICV_HASH "879741a7946b814455eee6c6ffde2984")
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES 86) # {x86, i386, i686}
    set(OPENCV_ICV_NAME "ippicv_2020_win_ia32_20191018_general.zip")
    set(OPENCV_ICV_HASH "cd39bdf0c2e1cac9a61101dad7a2413e")
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(OPENCV_ICV_NAME "ippicv_2020_lnx_intel64_20191018_general.tgz")
    set(OPENCV_ICV_HASH "7421de0095c7a39162ae13a6098782f9")
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES 86) # {x86, i386, i686}
    set(OPENCV_ICV_NAME "ippicv_2020_lnx_ia32_20191018_general.tgz")
    set(OPENCV_ICV_HASH "ad189a940fb60eb71f291321322fe3e8")
else()
    set(WITH_IPPICV OFF)
    message(FATAL_ERROR "IPP-ICV disabled: Unsupported Platform.")
    return()
endif()

set_local_or_remote_url(
    IPPICV_URL
    LOCAL_URL "${THIRD_PARTY_DOWNLOAD_DIR}/${OPENCV_ICV_NAME}"
    REMOTE_URLS "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/${OPENCV_ICV_NAME}"
    )

ExternalProject_Add(ext_ippicv
    PREFIX ippicv
    URL "${IPPICV_URL}"
    URL_HASH MD5=${OPENCV_ICV_HASH}
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ippicv/CMakeLists.txt <SOURCE_DIR>
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

ExternalProject_Get_Property(ext_ippicv INSTALL_DIR)
set(IPPICV_INCLUDE_DIR "${INSTALL_DIR}/include/icv/" "${INSTALL_DIR}/include/")
if (WIN32)
    set(IPPICV_LIBRARIES ippiw ippicvmt)
else ()
    set(IPPICV_LIBRARIES ippiw ippicv)
endif ()
set(IPPICV_LIB_DIR "${INSTALL_DIR}/lib")
set(IPPICV_VERSION_STRING "2020.0.0 Gold")  # From icv/ippversion.h

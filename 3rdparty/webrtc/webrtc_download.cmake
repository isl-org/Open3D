include(ExternalProject)

set(WEBRTC_VER 60e6748)
if (APPLE)
    set_local_or_remote_url(WEBRTC_URL
        LOCAL_URL
        "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/webrtc_${WEBRTC_VER}_macos_10.14.tar.gz"
        REMOTE_URLS
        https://github.com/intel-isl/open3d_downloads/releases/download/webrtc/webrtc_${WEBRTC_VER}_macos_10.14.tar.gz)
    set(WEBRTC_SHA256 e9d1f4e4fefb2e28ef4f16cf4a4f0008baf4fe638ca3ad329e82e7fd0ce87f56)
elseif (WIN32)
    set_local_or_remote_url(WEBRTC_URL
        LOCAL_URL "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/webrtc_${WEBRTC_VER}_win.zip"
        REMOTE_URLS https://github.com/intel-isl/open3d_downloads/releases/download/webrtc/webrtc_${WEBRTC_VER}_win.zip)
    set(WEBRTC_SHA256 f4686d0028ef5c36c5d7158a638fa834b63183b522f0b63932f7f70ebffeea22)
else()  # Linux
    if(GLIBCXX_USE_CXX11_ABI)
        set_local_or_remote_url(WEBRTC_URL
        LOCAL_URL "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/webrtc_${WEBRTC_VER}_cxx-abi-1.tar.gz"
        REMOTE_URLS https://github.com/intel-isl/open3d_downloads/releases/download/webrtc-v3/webrtc_${WEBRTC_VER}_cxx-abi-1.tar.gz)
        set(WEBRTC_SHA256 0d98ddbc4164b9e7bfc50b7d4eaa912a753dabde0847d85a64f93a062ae4c335)
    else()
        set_local_or_remote_url(WEBRTC_URL
        LOCAL_URL "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/webrtc_${WEBRTC_VER}_cxx-abi-0.tar.gz"
        REMOTE_URLS https://github.com/intel-isl/open3d_downloads/releases/download/webrtc-v3/webrtc_${WEBRTC_VER}_cxx-abi-0.tar.gz)
        set(WEBRTC_SHA256 2a3714713908f84079f1fbce8594c9b7010846b5db74b086f7bf30f22f1f5835)
    endif()
endif()

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    URL ${WEBRTC_URL}
    URL_HASH SHA256=${WEBRTC_SHA256}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ""
)

ExternalProject_Get_Property(ext_webrtc SOURCE_DIR)
if (WIN32)
    set(SOURCE_DIR "${SOURCE_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>")
endif()
set(LIBPNG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBPNG_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(LIBPNG_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)

# Variables consumed by find_dependencies.cmake
set(WEBRTC_INCLUDE_DIRS
    ${SOURCE_DIR}/include/
    ${SOURCE_DIR}/include/third_party/abseil-cpp/
    ${SOURCE_DIR}/include/third_party/jsoncpp/source/include/
    ${SOURCE_DIR}/include/third_party/jsoncpp/generated/
    ${SOURCE_DIR}/include/third_party/libyuv/include/
)
set(WEBRTC_LIB_DIR ${SOURCE_DIR}/lib)
set(WEBRTC_LIBRARIES
    webrtc
    webrtc_extra
)

# Dummy target that depends on all WebRTC targets.
add_custom_target(ext_webrtc_all)
add_dependencies(ext_webrtc_all ext_webrtc)

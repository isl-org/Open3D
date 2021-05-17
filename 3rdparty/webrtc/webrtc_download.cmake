include(ExternalProject)

if(GLIBCXX_USE_CXX11_ABI)
    set(WEBRTC_URL https://github.com/intel-isl/open3d_downloads/releases/download/webrtc-v2/webrtc_60e6748_cxx-abi-1.tar.gz)
    set(WEBRTC_SHA256 132956d1711fdaad281e017ebda077571c047ed323a1ab025c2be9e2aef2f973)
else()
    set(WEBRTC_URL https://github.com/intel-isl/open3d_downloads/releases/download/webrtc-v2/webrtc_60e6748_cxx-abi-0.tar.gz)
    set(WEBRTC_SHA256 64de92d7e5cadd8d8e5509b86880a13a2e8699dde6517696a81425535e09a8a9)
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

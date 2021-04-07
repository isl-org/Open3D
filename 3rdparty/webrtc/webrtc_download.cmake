include(ExternalProject)

set(WEBRTC_URL ${PROJECT_SOURCE_DIR}/webrtc_e2ac591.tar.gz)
# set(WEBRTC_SHA256 a4eec51612048001a2b3e08de9377a80934ed43aee56423a70a19aae73d4edef)

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    URL ${WEBRTC_URL}
    # URL_HASH SHA256=${WEBRTC_SHA256}
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

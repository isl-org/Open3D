include(ExternalProject)

ExternalProject_Add(
        ext_msgpack-c
        PREFIX msgpack-c
        # msgpack-cxx >= 5.0 fixes config for MSVC's standards-conforming
        # preprocessor (/Zc:preprocessor), which is required when building
        # against CUDA 13.2 CCCL headers. The 3.3.0 release fails to expand
        # MSGPACK_DEFINE_MAP under that preprocessor mode.
        URL https://github.com/msgpack/msgpack-c/releases/download/cpp-7.0.0/msgpack-cxx-7.0.0.tar.gz
        URL_HASH SHA256=7504b7af7e7b9002ce529d4f941e1b7fb1fb435768780ce7da4abaac79bb156f
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/msgpack"
        # do not configure
        CONFIGURE_COMMAND ""
        # do not build
        BUILD_COMMAND ""
        # do not install
        INSTALL_COMMAND ""
        )
ExternalProject_Get_Property( ext_msgpack-c SOURCE_DIR )
set( MSGPACK_INCLUDE_DIRS "${SOURCE_DIR}/include/" )

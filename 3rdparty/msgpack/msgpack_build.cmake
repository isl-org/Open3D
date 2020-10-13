include(ExternalProject)

ExternalProject_Add(
        ext_msgpack-c
        PREFIX msgpack-c
        URL https://github.com/msgpack/msgpack-c/releases/download/cpp-3.3.0/msgpack-3.3.0.tar.gz
        URL_HASH MD5=e676575d52caae974e579c3d5f0ba6a2
        # do not configure
        CONFIGURE_COMMAND ""
        # do not build
        BUILD_COMMAND ""
        # do not install
        INSTALL_COMMAND ""
        )
ExternalProject_Get_Property( ext_msgpack-c SOURCE_DIR )
set( MSGPACK_INCLUDE_DIRS "${SOURCE_DIR}/include/" )

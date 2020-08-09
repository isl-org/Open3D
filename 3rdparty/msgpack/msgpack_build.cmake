include(FetchContent)

FetchContent_Declare(
    ext_msgpack-c
    URL https://github.com/msgpack/msgpack-c/releases/download/cpp-3.3.0/msgpack-3.3.0.tar.gz
    URL_HASH MD5=e676575d52caae974e579c3d5f0ba6a2
)
FetchContent_GetProperties(msgpack-c)
if(NOT ext_msgpack-c_POPULATED)
    FetchContent_Populate(ext_msgpack-c)
    # just download and treat as header only lib
    set(MSGPACK_INCLUDE_DIRS "${ext_msgpack-c_SOURCE_DIR}/include/")
endif()


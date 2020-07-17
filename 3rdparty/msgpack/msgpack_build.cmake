include(FetchContent)

FetchContent_Declare(
    ext_msgpack-c
    GIT_REPOSITORY "https://github.com/msgpack/msgpack-c.git"
    GIT_TAG da2fc25f875a0394e2eaa4ddfa0fc9221b6b4c52
    )
FetchContent_GetProperties(msgpack-c)
if(NOT ext_msgpack-c_POPULATED)
    FetchContent_Populate(ext_msgpack-c)
    # just download and treat as header only lib
    set(MSGPACK_INCLUDE_DIRS "${ext_msgpack-c_SOURCE_DIR}/include/")
endif()


include(ExternalProject)

ExternalProject_Add(
    ext_jsoncpp
    PREFIX jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
    GIT_TAG 1.9.4
    GIT_SHALLOW ON  # Do not download the history.
    UPDATE_COMMAND ""
    PATCH_COMMAND git apply ${Open3D_3RDPARTY_DIR}/jsoncpp/0001-optional-CXX11-ABI-and-MSVC-runtime.patch
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_STATIC_LIBS=ON
        -DBUILD_OBJECT_LIBS=OFF
        -DJSONCPP_WITH_TESTS=OFF
        -DJSONCPP_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
        -DJSONCPP_STATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
    BUILD_ALWAYS ON
)

ExternalProject_Get_Property(ext_jsoncpp INSTALL_DIR)
set(JSONCPP_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(JSONCPP_LIB_DIR ${INSTALL_DIR}/lib)
set(JSONCPP_LIBRARIES jsoncpp)

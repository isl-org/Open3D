include(ExternalProject)

if(GLIBCXX_USE_CXX11_ABI)
    set(JSONCPP_CXX_ABI 1)
else()
    set(JSONCPP_CXX_ABI 0)
endif()

ExternalProject_Add(
    ext_jsoncpp
    PREFIX jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
    GIT_TAG 1.9.4
    UPDATE_COMMAND ""
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
        -DJSONCPP_CXX_ABI=${JSONCPP_CXX_ABI}
    BUILD_ALWAYS ON
)

ExternalProject_Get_Property(ext_jsoncpp INSTALL_DIR)
set(JSONCPP_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(JSONCPP_LIB_DIR ${INSTALL_DIR}/lib)
set(JSONCPP_LIBRARIES jsoncpp)

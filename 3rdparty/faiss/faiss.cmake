include(ExternalProject)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
set(BLAS_LIB ${MKL_LIBRARIES})

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

ExternalProject_Add(
    ext_faiss
    PREFIX faiss
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/faiss/faiss
    CONFIGURE_COMMAND LDFLAGS=-L${MKL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/faiss/faiss/configure --without-cuda --prefix=${3RDPARTY_INSTALL_PREFIX}
    BUILD_COMMAND ${MAKE}
    BUILD_IN_SOURCE 1
)
add_library(faiss INTERFACE)
add_dependencies(faiss ext_faiss)

if(MSVC)
    set(lib_name "faiss-static")
else()
    set(lib_name "faiss")
endif()

set(FAISS_LIBRARIES ${lib_name})
set(faiss_LIB_FILES ${3RDPARTY_INSTALL_PREFIX}/${LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
target_include_directories(faiss SYSTEM INTERFACE ${3RDPARTY_INSTALL_PREDIX}/include)
target_link_libraries(faiss INTERFACE ${faiss_LIB_FILES})
add_dependencies(build_all_3rd_party_libs faiss)
    
if (NOT BUILD_SHARED_LIBS)
    install(FILES ${faiss_LIB_FILES}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

add_dependencies(build_all_3rd_party_libs faiss)



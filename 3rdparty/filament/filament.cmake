include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

# TODO: don't build samples

set(FILAMENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/filament/filament)
set(FILAMENT_TMP_INSTALL_DIR ${CMAKE_BINARY_DIR}/filament-install)
set(FILAMENT_TMP_LIB_DIR ${FILAMENT_TMP_INSTALL_DIR}/lib/x86_64)

ExternalProject_Add(
    ext_filament
    PREFIX filament
    SOURCE_DIR ${FILAMENT_SOURCE_DIR}
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_TMP_INSTALL_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

set(filament_LIB_FILES libfilameshio.a libfilament.a libfilamat_lite.a libfilaflat.a libfilabridge.a libgeometry.a libbackend.a libbluegl.a libbluevk.a libibl.a libimage.a libmeshoptimizer.a libsmol-v.a libutils.a)

set(filament_INCLUDE_DIRS ${3RDPARTY_INSTALL_PREFIX}/include/filament)
add_custom_target(filament_copy
    COMMAND cp -a include ${filament_INCLUDE_DIRS}
    COMMAND cp -a lib/${CMAKE_SYSTEM_PROCESSOR}/* ${3RDPARTY_INSTALL_PREFIX}/lib
    COMMAND cp -a bin/* ${3RDPARTY_INSTALL_PREFIX}/bin
    WORKING_DIRECTORY ${FILAMENT_TMP_INSTALL_DIR}
    DEPENDS ext_filament
    )
add_library(filament INTERFACE)
add_dependencies(filament filament_copy)

## If MSVC, the OUTPUT_NAME was set to filament-static
#if(MSVC)
#    set(lib_name "filament-static")
#else()
#    set(lib_name "filament")
#endif()

# For linking with Open3D after installation
set(filament_LIBRARIES ${filament_LIB_FILES} ${SDL2_LIBRARIES} ${FREETYPE_LIBRARIES})

#target_include_directories(filament SYSTEM INTERFACE
target_include_directories(filament SYSTEM INTERFACE
    ${3RDPARTY_INSTALL_PREFIX}/include/filament
)
#target_link_libraries(filament INTERFACE ${FILAMENT_LIBRARIES})

#if (NOT BUILD_SHARED_LIBS)
#    install(FILES ${filament_LIB_FILES} libfilament_all.a
#            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
#endif()

add_dependencies(build_all_3rd_party_libs filament)

if (WIN32)
elseif (APPLE)
    find_library(CORE_VIDEO CoreVideo)
    find_library(QUARTZ_CORE QuartzCore)
    find_library(OPENGL_LIBRARY OpenGL)
    find_library(METAL_LIBRARY Metal)
    find_library(APPKIT_LIBRARY AppKit)
    list(APPEND filament_LIBRARIES ${CORE_VIDEO} ${QUARTZ_CORE} ${OPENGL_LIBRARY} ${METAL_LIBRARY} ${APPKIT_LIBRARY})

    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fobjc-link-runtime")
else ()
#    # These are needed by Clang on Linux
#    target_link_libraries(demo pthread)
#    target_link_libraries(demo dl)
#    target_link_libraries(demo c++)
#    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

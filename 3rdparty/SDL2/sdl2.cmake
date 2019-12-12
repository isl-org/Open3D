include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

ExternalProject_Add(
    ext_sdl2
    PREFIX sdl2
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDL2/SDL2-2.0.10
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_INSTALL_PREFIX=${3RDPARTY_INSTALL_PREFIX}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DFORCE_STATIC_VCRT=${STATIC_WINDOWS_RUNTIME}
#       -DVIDEO_VULKAN=${USE_VULKAN}
)


add_library(sdl2_combined INTERFACE)
if (WIN32)
    target_link_libraries(sdl2_combined INTERFACE SDL2$<$<CONFIG:Debug>:d>) #SDL2main$<$<CONFIG:Debug>:d>)
else ()
    target_link_libraries(sdl2_combined INTERFACE SDL2) #SDL2main)
endif()

#target_include_directories(sdl2 SYSTEM INTERFACE
#    ${3RDPARTY_INSTALL_PREFIX}/include/sdl2
#)

set(SDL2_LIBRARIES sdl2_combined)

add_dependencies(build_all_3rd_party_libs sdl2_combined)
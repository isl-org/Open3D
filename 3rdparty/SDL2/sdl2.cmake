include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

set(SDL_TMP_INSTALL_DIR ${CMAKE_BINARY_DIR}/sdl-install)

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
        -DCMAKE_INSTALL_PREFIX=${SDL_TMP_INSTALL_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DFORCE_STATIC_VCRT=${STATIC_WINDOWS_RUNTIME}
#       -DVIDEO_VULKAN=${USE_VULKAN}
)

if (WIN32)
    add_custom_target(sdl_copy
        COMMAND xcopy /s /i /y /q \"include\" \"${3RDPARTY_INSTALL_PREFIX}/include\"
        COMMAND xcopy /s /i /y /q \"lib\" \"${3RDPARTY_INSTALL_PREFIX}/lib\"
        # Binaries go to examples directory
        COMMAND xcopy /s /i /y /q \"bin\" \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/${CMAKE_BUILD_TYPE}\" 
        WORKING_DIRECTORY ${SDL_TMP_INSTALL_DIR}
        DEPENDS ext_sdl2
        )
else()
    add_custom_target(sdl_copy
        COMMAND cp -a include ${3RDPARTY_INSTALL_PREFIX}/include
        COMMAND cp -a lib/* ${3RDPARTY_INSTALL_PREFIX}/lib
        WORKING_DIRECTORY ${SDL_TMP_INSTALL_DIR}
        DEPENDS ext_sdl2
        )
endif()

add_library(sdl2_combined INTERFACE)
add_dependencies(sdl2_combined sdl_copy)

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
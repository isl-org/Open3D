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
        COMMAND cp -a include/* ${3RDPARTY_INSTALL_PREFIX}/include
        COMMAND cp -a lib/* ${3RDPARTY_INSTALL_PREFIX}/lib
        WORKING_DIRECTORY ${SDL_TMP_INSTALL_DIR}
        DEPENDS ext_sdl2
        )
endif()

add_library(sdl2_combined INTERFACE)
add_dependencies(sdl2_combined sdl_copy)

if (WIN32)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" LOWER_BUILD_TYPE)
    if (LOWER_BUILD_TYPE MATCHES debug)
        set(DEBUG_SUFFIX "d")
     else()
        set(DEBUG_SUFFIX "")
     endif()
else()
        set(DEBUG_SUFFIX "")
endif()
set(SDL2_LIB_FILES ${3RDPARTY_INSTALL_PREFIX}/${LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}SDL2${DEBUG_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(sdl2_combined INTERFACE ${SDL2_LIB_FILES})

#target_include_directories(sdl2 SYSTEM INTERFACE
#    ${3RDPARTY_INSTALL_PREFIX}/include/sdl2
#)

set(SDL2_LIBRARIES ${SDL2_LIB_FILES})

add_dependencies(build_all_3rd_party_libs sdl2_combined)

if (APPLE)
    find_library(CORE_AUDIO CoreAudio)
    find_library(AUDIO_TOOLBOX AudioToolbox)
    find_library(FORCE_FEEDBACK ForceFeedback)
    find_library(CARBON Carbon)
    # the system libiconv uses different symbol names than MacPorts or Brew,
    # so specify the system libiconv. find_library() seems to prefer the
    # non-system libraries
    list(APPEND SDL2_LIBRARIES ${CORE_AUDIO} ${AUDIO_TOOLBOX} ${FORCE_FEEDBACK} ${CARBON} /usr/lib/libiconv.dylib)
endif()

if (NOT BUILD_SHARED_LIBS)
    install(FILES ${SDL2_LIB_FILES}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

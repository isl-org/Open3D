# If user not provided path to filament
if ("${PATH_TO_FILAMENT}" STREQUAL "")
    set(FILAMENT_ROOT ${CMAKE_BINARY_DIR}/downloads/filament)

    if (USE_VUKLAN AND (ANDROID OR WIN32 OR WEBGL OR IOS))
        MESSAGE(FATAL_ERROR "Downloadable version of Filament supports vulkan only on Linux and Apple")
    endif()

    if (NOT EXISTS ${FILAMENT_ROOT}/README.md)
        set(DOWNLOAD_PATH ${CMAKE_BINARY_DIR}/downloads)
        set(TAR_PWD ${DOWNLOAD_PATH})

        if (NOT EXISTS ${ARCHIVE_FILE})
            set(ARCHIVE_FILE ${CMAKE_BINARY_DIR}/downloads/filament.tgz)

            # Setup download links ============================================================================
            set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200220-linux.tgz")
            set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-linux.tgz")

            if (WIN32)
                set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-windows.tgz")
                set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-windows.tgz")
                
                file(MAKE_DIRECTORY ${FILAMENT_ROOT})
                set(TAR_PWD ${FILAMENT_ROOT})
            elseif (APPLE)
                set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-mac-10.14-resizefix2.tgz")
                set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-mac.tgz")
            endif()
            # =================================================================================================

            file(DOWNLOAD ${DOWNLOAD_URL_PRIMARY} ${ARCHIVE_FILE} SHOW_PROGRESS STATUS DOWNLOAD_RESULT)
            if (NOT DOWNLOAD_RESULT EQUAL 0)
                file(DOWNLOAD ${DOWNLOAD_URL_FALLBACK} ${ARCHIVE_FILE} SHOW_PROGRESS STATUS DOWNLOAD_RESULT)
            endif()
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${ARCHIVE_FILE} WORKING_DIRECTORY ${TAR_PWD})
    endif()
else()
    set(FILAMENT_ROOT ${PATH_TO_FILAMENT})
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")

set(filament_INCLUDE_DIRS ${3RDPARTY_INSTALL_PREFIX}/include/filament)
set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl ibl image meshoptimizer smol-v utils)
if (UNIX)
    set(filament_LIBRARIES ${filament_LIBRARIES} bluevk)
endif()

file(MAKE_DIRECTORY ${filament_INCLUDE_DIRS})
file(MAKE_DIRECTORY ${3RDPARTY_INSTALL_PREFIX}/bin)
file(MAKE_DIRECTORY ${3RDPARTY_INSTALL_PREFIX}/lib)

# Copy necessary files to our 3rdparty install folder
if (WIN32)
    if (STATIC_WINDOWS_RUNTIME)
        set(CRT_CONFIG mt)
    else()
        set(CRT_CONFIG md)
    endif()

    link_directories("${3RDPARTY_INSTALL_PREFIX}/lib/$(Configuration)")

    add_custom_target(filament_copy
            COMMAND xcopy /d /s /i /y /q \"include\" \"${filament_INCLUDE_DIRS}\"
            COMMAND xcopy /d /s /i /y /q \"lib/x86_64/${CRT_CONFIG}\" \"${3RDPARTY_INSTALL_PREFIX}/lib/Release\"
            COMMAND xcopy /d /s /i /y /q \"lib/x86_64/${CRT_CONFIG}d\" \"${3RDPARTY_INSTALL_PREFIX}/lib/Debug\"
            COMMAND xcopy /d /s /i /y /q \"bin\" \"${3RDPARTY_INSTALL_PREFIX}/bin\"
            WORKING_DIRECTORY ${FILAMENT_ROOT})

    if (CMAKE_BUILD_TYPE MATCHES DEBUG)
        set(FILAMENT_LIB_SRC_PATH ${FILAMENT_ROOT}/lib/x86_64/${CRT_CONFIG}d)
    else ()
        set(FILAMENT_LIB_SRC_PATH ${FILAMENT_ROOT}/lib/x86_64/${CRT_CONFIG})
    endif()
else()
    add_custom_target(filament_copy
            COMMAND cp -a include/* ${filament_INCLUDE_DIRS}
            COMMAND cp -a lib/${CMAKE_SYSTEM_PROCESSOR}/* ${3RDPARTY_INSTALL_PREFIX}/lib
            COMMAND cp -a bin/* ${3RDPARTY_INSTALL_PREFIX}/bin
            WORKING_DIRECTORY ${FILAMENT_ROOT})

    set(FILAMENT_LIB_SRC_PATH ${FILAMENT_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR})
endif()

add_dependencies(build_all_3rd_party_libs filament_copy)

if (NOT BUILD_SHARED_LIBS)
    install(DIRECTORY "${FILAMENT_LIB_SRC_PATH}/"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
endif()

install(DIRECTORY "${filament_INCLUDE_DIRS}"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/3rdparty"
            PATTERN     "*.c"           EXCLUDE
            PATTERN     "*.cmake"       EXCLUDE
            PATTERN     "*.cpp"         EXCLUDE
            PATTERN     "*.in"          EXCLUDE
            PATTERN     "*.m"           EXCLUDE
            PATTERN     "*.txt"         EXCLUDE
            PATTERN     ".gitignore"    EXCLUDE)

if (WIN32)
elseif (APPLE)
    find_library(CORE_VIDEO CoreVideo)
    find_library(QUARTZ_CORE QuartzCore)
    find_library(OPENGL_LIBRARY OpenGL)
    find_library(METAL_LIBRARY Metal)
    find_library(APPKIT_LIBRARY AppKit)
    list(APPEND filament_LIBRARIES ${CORE_VIDEO} ${QUARTZ_CORE} ${OPENGL_LIBRARY} ${METAL_LIBRARY} ${APPKIT_LIBRARY})

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fobjc-link-runtime")
else ()
    # These are needed by Clang on Linux
    list(APPEND filament_LIBRARIES pthread dl c++)
endif()

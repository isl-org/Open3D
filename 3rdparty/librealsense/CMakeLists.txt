project(realsense C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(REALSENSE_LIBRARY realsense)

file(GLOB librealsense_sources src/*.c src/*.cpp)
file(GLOB librealsense_headers src/*.h)
file(GLOB libuvc_sources src/libuvc/*.c)
file(GLOB libuvc_headers src/libuvc/*.h)

if(WIN32)
    set(realsense_sources ${librealsense_sources})
    set(realsense_headers ${librealsense_headers})
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DRS_USE_WMF_BACKEND)
    if(MSVC)
        # suppress warnings
		add_definitions(/wd4267 /wd4005 /wd4244 /wd4838 /wd4311 /wd4302 /wd4819)
	endif(MSVC)
elseif(CYGWIN)
    set(realsense_sources ${librealsense_sources})
    set(realsense_headers ${librealsense_headers})
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DRS_USE_WMF_BACKEND)
elseif(APPLE)
    set(realsense_sources ${librealsense_sources} ${libuvc_sources})
    set(realsense_headers ${librealsense_headers} ${libuvc_headers})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pedantic -msse4.1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pedantic -msse4.1")
    find_package(libusb-1.0 REQUIRED)
    add_definitions(-DRS_USE_LIBUVC_BACKEND)
    include_directories(
    	${LIBUSB_1_INCLUDE_DIRS}
    	)
    # suppress some warnings
    add_compile_options(-Wno-constant-logical-operand)
    add_compile_options(-Wno-braced-scalar-init)
    add_compile_options(-Wno-four-char-constants)
    add_compile_options(-Wno-zero-length-array)
    add_compile_options(-Wno-newline-eof)
    add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)
    add_compile_options(-Wno-pedantic)
    add_compile_options(-Wno-multichar)
    set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
elseif(UNIX)
endif(WIN32)

add_library(${REALSENSE_LIBRARY} STATIC ${realsense_sources} ${realsense_headers})

if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
    target_link_libraries(${REALSENSE_LIBRARY} ${LIBUSB_1_LIBRARIES})
elseif(UNIX)
endif(WIN32)

set_target_properties(${REALSENSE_LIBRARY} PROPERTIES
        OUTPUT_NAME ${REALSENSE_LIBRARY}
        POSITION_INDEPENDENT_CODE ON
        FOLDER "3rdparty"
    )

# install
#install(TARGETS ${REALSENSE_LIBRARY}
#        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(REALSENSE_LIBRARY "${REALSENSE_LIBRARY}" PARENT_SCOPE)

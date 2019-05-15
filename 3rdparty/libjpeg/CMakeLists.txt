project(libjpeg C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(JPEG_LIBRARY jpeg)

file(GLOB libjpeg_sources *.c)
file(GLOB libjpeg_headers *.h)

if(WIN32)
	if(MSVC)
		add_definitions(/wd4013)
		add_definitions(/wd4267)
		set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /ignore:4006")
	endif(MSVC)
elseif(CYGWIN)
elseif(APPLE)
	add_compile_options(-Wno-implicit-function-declaration)
elseif(UNIX)
endif(WIN32)

add_library(${JPEG_LIBRARY} STATIC ${libjpeg_sources} ${libjpeg_headers})
set_target_properties(${JPEG_LIBRARY} PROPERTIES
        OUTPUT_NAME ${JPEG_LIBRARY}
        FOLDER "3rdparty"
    )

set(JPEG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(JPEG_LIBRARIES "jpeg")
set(JPEG_INCLUDE_DIR ${JPEG_INCLUDE_DIR} PARENT_SCOPE)
set(JPEG_LIBRARIES ${JPEG_LIBRARIES} PARENT_SCOPE)

# install
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${JPEG_LIBRARY}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

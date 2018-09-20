project(tinyfiledialogs C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(FILEDIALOG_LIBRARY tinyfiledialogs)

file(GLOB tinyfiledialogs_sources tinyfiledialogs.c)
file(GLOB tinyfiledialogs_headers tinyfiledialogs.h)

add_library(${FILEDIALOG_LIBRARY} STATIC ${tinyfiledialogs_sources} ${tinyfiledialogs_headers})
set_target_properties(${FILEDIALOG_LIBRARY} PROPERTIES
        OUTPUT_NAME ${FILEDIALOG_LIBRARY}
        FOLDER "3rdparty"
    )

# install
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${FILEDIALOG_LIBRARY}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

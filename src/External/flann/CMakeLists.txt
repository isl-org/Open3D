project(flann)

cmake_minimum_required(VERSION 3.0.0)
set(FLANN_LIBRARY flann)

# suppress warnings
if (WIN32)
    # lots of warnings with cl.exe right now, use /W1
    add_definitions("/W1 -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /bigobj")
else(WIN32)
    add_definitions( "-Wall -Wno-unknown-pragmas -Wno-unused-function" )
endif(WIN32)

file(GLOB CPP_SOURCES flann_cpp.cpp)
file(GLOB CPP_HEADERS *.h *.hpp algorithms/*.h util/*.h nn/*.h)

add_library(${FLANN_LIBRARY} STATIC ${CPP_SOURCES} ${CPP_HEADERS})
set_target_properties(${FLANN_LIBRARY} PROPERTIES
        OUTPUT_NAME ${FLANN_LIBRARY}
        FOLDER "External"
    )

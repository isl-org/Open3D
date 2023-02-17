# Adapted from https://github.com/opencv/opencv/blob/master/3rdparty/ippicv/CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(ippiw)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
file(GLOB lib_srcs iw/src/*.c)
file(GLOB lib_hdrs iw/include/*.h iw/include/iw/*.h iw/include/iw++/*.hpp)
file(GLOB IPPICV_LIBRARY
    icv/lib/*/${CMAKE_STATIC_LIBRARY_PREFIX}*${CMAKE_STATIC_LIBRARY_SUFFIX})

add_library(ippiw STATIC ${lib_srcs} ${lib_hdrs})
target_compile_definitions(ippiw PRIVATE IW_BUILD ICV_BASE)
target_include_directories(ippiw PRIVATE icv/include iw/include)
target_compile_definitions(ippiw PRIVATE IW_ENABLE_iwiResize_Nearest=1)

if(UNIX)
    if(${CMAKE_C_COMPILER_ID} MATCHES GNU OR
            ${CMAKE_C_COMPILER_ID} MATCHES Clang OR
            ${CMAKE_C_COMPILER_ID} MATCHES Intel)
        target_compile_options(ippiw PRIVATE -Wno-unused-function -Wno-missing-braces -Wno-missing-field-initializers)
  endif()
  if(${CMAKE_C_COMPILER_ID} MATCHES Clang)
    target_compile_options(ippiw PRIVATE -Wno-self-assign)
  endif()
endif()

install(TARGETS ippiw
    ARCHIVE DESTINATION lib)
install(FILES ${IPPICV_LIBRARY}
    DESTINATION lib)
install(DIRECTORY iw/include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY icv/include/ DESTINATION include/icv
    FILES_MATCHING PATTERN "*.h*")

set(LIVE ${CMAKE_BINARY_DIR}/live)

FILE(GLOB LIVEHELPERSOURCE
    ${LIVE}/groupsock/*.c*
    ${LIVE}/liveMedia/*.c*
    ${LIVE}/UsageEnvironment/*.c*
    ${LIVE}/BasicUsageEnvironment/*.c*
    src/*.cpp
)
add_library (liblive555helper STATIC ${LIVEHELPERSOURCE})
target_include_directories(liblive555helper PUBLIC
    ${LIVE}/groupsock/include
    ${LIVE}/liveMedia/include
    ${LIVE}/UsageEnvironment/include
    ${LIVE}/BasicUsageEnvironment/include
    inc
)

add_definitions(
    -DNO_OPENSSL=1
    -DBSD=1
    -DSOCKLEN_T=socklen_t
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE_SOURCE=1
)

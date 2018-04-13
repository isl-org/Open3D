file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(EncodeShader ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(EncodeShader PROPERTIES
        FOLDER "Tools"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS EncodeShader
        RUNTIME DESTINATION bin)

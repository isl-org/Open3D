file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestRegistrationRANSAC ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TestRegistrationRANSAC Visualization IO Core)
set_target_properties(TestRegistrationRANSAC PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestRegistrationRANSAC
        RUNTIME DESTINATION test/bin)

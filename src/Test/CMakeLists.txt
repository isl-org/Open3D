add_executable(TestPointCloud TestPointCloud.cpp)
target_link_libraries(TestPointCloud Core IO)

#file(GLOB TEST_DATA_FILES "TestData/*.*")
#file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/TestData)
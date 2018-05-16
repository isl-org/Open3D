macro(EXPERIMENT EXPERIMENT_NAME)
    add_executable(${EXPERIMENT_NAME} "${EXPERIMENT_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXPERIMENT_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXPERIMENT_NAME} PROPERTIES
            FOLDER "EXPERIMENTAL"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
endmacro(EXPERIMENT)

EXPERIMENT(EvaluateFeatureMatch      ${PROJECT_NAME})
EXPERIMENT(EvaluatePCDMatch          ${PROJECT_NAME})
EXPERIMENT(IntegrateRGBD             ${PROJECT_NAME})
EXPERIMENT(OdometryRGBD              ${PROJECT_NAME})
EXPERIMENT(TrimMeshBasedOnPointCloud ${PROJECT_NAME})
EXPERIMENT(ViewDistances             ${PROJECT_NAME})
EXPERIMENT(ViewPCDMatch              ${PROJECT_NAME})

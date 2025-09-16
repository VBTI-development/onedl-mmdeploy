# Copyright (c) OpenMMLab. All rights reserved.

project(tests)

# Common sources and setup
set(TC_SRC test_main.cpp)


# Function to create individual tests
function(add_mmdeploy_test test_name test_sources)
    add_executable(${test_name} ${TC_SRC} ${test_sources})
    target_include_directories(${test_name}
            PRIVATE ${CMAKE_SOURCE_DIR}/third_party/catch2)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (NOT (MMDEPLOY_SHARED_LIBS OR MSVC))
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    endif ()

    mmdeploy_load_static(${test_name} MMDeployStaticModules)
    mmdeploy_load_dynamic(${test_name} MMDeployDynamicModules)
    target_link_libraries(${test_name} PRIVATE
        MMDeployLibs
        mmdeploy_transform
        mmdeploy_operation
        mmdeploy_opencv_utils)

    # Add the test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Generate the header file
configure_file(config/test_define.h.in
        ${CMAKE_CURRENT_SOURCE_DIR}/test_define.h)

# Add individual test categories
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/archive ARCHIVE_TC)
if(ARCHIVE_TC)
    add_mmdeploy_test(mmdeploy_archive_tests "${ARCHIVE_TC}")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core CORE_TC)
if(CORE_TC)
    add_mmdeploy_test(mmdeploy_core_tests "${CORE_TC}")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/preprocess TRANSFORM_TC)
if(TRANSFORM_TC)
    add_mmdeploy_test(mmdeploy_preprocess_tests "${TRANSFORM_TC}")

    # Create test resources directory and copy test files
    set(TEST_RESOURCES_DIR ${CMAKE_CURRENT_BINARY_DIR}/mmdeploy_test_resources)
    set(TRANSFORM_RESOURCES_DIR ${TEST_RESOURCES_DIR}/transform)

    # Create the directories
    file(MAKE_DIRECTORY ${TRANSFORM_RESOURCES_DIR})

    # Copy test image if it exists
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/data/tiger.jpeg)
        configure_file(
            ${CMAKE_SOURCE_DIR}/tests/data/tiger.jpeg
            ${TRANSFORM_RESOURCES_DIR}/tiger.jpeg
            COPYONLY
        )
        message(STATUS "Copied tiger.jpeg to test resources directory")
    else()
        message(WARNING "Test image tiger.jpeg not found at ${CMAKE_SOURCE_DIR}/tests/data/tiger.jpeg")
    endif()
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/net NET_TC)
if(NET_TC)
    add_mmdeploy_test(mmdeploy_net_tests "${NET_TC}")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/model MODEL_TC)
if(MODEL_TC)
    add_mmdeploy_test(mmdeploy_model_tests "${MODEL_TC}")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/graph GRAPH_TC)
if(GRAPH_TC)
    add_mmdeploy_test(mmdeploy_graph_tests "${GRAPH_TC}")
endif()

# Device tests
foreach (DEVICE IN LISTS MMDEPLOY_TARGET_DEVICES)
    set(DEVICE_TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/device/test_${DEVICE}_device.cpp)
    if(EXISTS ${DEVICE_TEST_FILE})
        add_mmdeploy_test(mmdeploy_${DEVICE}_device_tests "${DEVICE_TEST_FILE}")
    endif()
endforeach ()

# CAPI tests
set(TASK_LIST "model")
set(CODEBASES "${MMDEPLOY_CODEBASES}")
if ("mmcls" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "classifier")
endif ()
if ("mmdet" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "detector")
endif ()
if ("mmseg" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "segmentor")
endif ()
if ("mmedit" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "restorer")
endif ()
if ("mmocr" IN_LIST MMDEPLOY_CODEBASES)
    list(APPEND TASK_LIST "text_detector")
    list(APPEND TASK_LIST "text_recognizer")
endif ()

foreach (TASK ${TASK_LIST})
    set(CAPI_TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/capi/test_${TASK}.cpp)
    if(EXISTS ${CAPI_TEST_FILE})
        add_mmdeploy_test(mmdeploy_capi_${TASK}_tests "${CAPI_TEST_FILE}")
    endif()
endforeach ()

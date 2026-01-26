function(generate_multiplier_test MULTIPLIER)
  message(STATUS "Generating test for ${MULTIPLIER}.")

  # Get multiplier info. # TODO: Replace string matching with target properties.
  set(MULTIPLIER_HEADER "${MULTIPLIER}.hpp")
  if("${MULTIPLIER}" MATCHES "^[cC]uda.*")
    set(TEST_TYPE "PAR_EXEC_CUDA_TEST") # TODO: Replace this name.
    set(SOURCE_FILE_EXT "cu")
  else()
    set(TEST_TYPE "CXX_TEST")
    set(SOURCE_FILE_EXT "cpp")
  endif()

  # Set test info.
  set(TEST_NAME "${MULTIPLIER}_test")
  set(TEST_FILE_NAME "${TEST_NAME}.${SOURCE_FILE_EXT}")
  set(TEST_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/generated/${TEST_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply_test.cpp.in"
    "${TEST_FILE_PATH}"
  )

  # Create the test.
  add_executable("${TEST_NAME}" "${TEST_FILE_PATH}")
  target_link_libraries("${TEST_NAME}"
    GTest::GTest
    GTest::Main
    "${MULTIPLIER}"
  )
  add_test(NAME "${TEST_NAME}" COMMAND "${TEST_NAME}")
endfunction()
get_property(LOCAL_MULTIPLIERS GLOBAL PROPERTY MULTIPLIERS)
foreach(MULTIPLIER IN LISTS LOCAL_MULTIPLIERS)
  generate_multiplier_test("${MULTIPLIER}")
endforeach()

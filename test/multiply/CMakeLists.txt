function(generate_multiplier_test MULTIPLIER)
  # Get multiplier info.
  set(MULTIPLIER_HEADER "${MULTIPLIER}.hpp")
  get_target_property(SOURCE_FILE_EXT "${MULTIPLIER}" SOURCE_FILE_EXT)
  get_target_property(MULTIPLIER_FAMILY "${MULTIPLIER}" MULTIPLIER_FAMILY)

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

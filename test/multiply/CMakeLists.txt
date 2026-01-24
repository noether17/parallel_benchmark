function(generate_multiplier_test MULTIPLIER)
  message(STATUS "MULTIPLIER: ${MULTIPLIER}")
  if("${MULTIPLIER}" MATCHES "cuda.*")
    set(TEST_TYPE "PAR_EXEC_CUDA_TEST") # TODO: Replace this name.
    set(SOURCE_FILE_EXT "cu")
  else()
    set(TEST_TYPE "CXX_TEST")
    set(SOURCE_FILE_EXT "cpp")
  endif()
  message(STATUS "SOURCE_FILE_EXT: ${SOURCE_FILE_EXT}")
  set(MULTIPLIER_HEADER "${MULTIPLIER}.hpp")
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

get_property(local_cxx_multipliers GLOBAL PROPERTY CXX_MULTIPLIERS)
foreach(multiplier IN LISTS local_cxx_multipliers)
  generate_multiplier_test(${multiplier})
endforeach()

get_property(TEMP_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
foreach(CUDA_MULTIPLIER IN LISTS TEMP_CUDA_MULTIPLIERS)
  generate_multiplier_test(${CUDA_MULTIPLIER})
endforeach()

#########
# ParExec
#########
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")

function(generate_par_exec_test PAR_EXEC_TYPE)
  get_par_exec_properties("${PAR_EXEC_TYPE}")

  # Set multiplier and test name.
  set(MULTIPLIER "${PAR_EXEC_TOKEN}_multiply")
  set(TEST_NAME "${MULTIPLIER}_test")

  # Configure test file.
  set(MULTIPLIER_HEADER "${MULTIPLIER}.${PAR_EXEC_HEADER_EXT}")
  if(PAR_EXEC_BASENAME MATCHES "Cuda")
    set(TEST_TYPE "PAR_EXEC_CUDA_TEST")
  else()
    set(TEST_TYPE "PAR_EXEC_CXX_TEST")
  endif()
  set(TEST_FILE_NAME "${TEST_NAME}.${PAR_EXEC_SOURCE_EXT}")
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

get_property(TEMP_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PAR_EXEC IN LISTS TEMP_PARALLEL_EXECUTORS)
  generate_par_exec_test("${PAR_EXEC}")
endforeach()

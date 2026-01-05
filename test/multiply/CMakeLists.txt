function(generate_multiply_test_for_function
         test_type test_name function_name)
  # Configured text
  if(test_type STREQUAL "HOST")
    set(TEST_HEADER MultiplyTest.hpp)
    set(TEST_FIXTURE MultiplyTest)
    set(OUT_FILE_EXT cpp)
  elseif(test_type STREQUAL "DEVICE")
    set(TEST_HEADER DeviceMultiplyTest.cuh)
    set(TEST_FIXTURE DeviceMultiplyTest)
    set(OUT_FILE_EXT cu)
  else()
    message(FATAL_ERROR "UNKNOWN TEST TYPE ${test_type}!")
  endif()
  set(TEST_NAME "${test_name}")
  set(FUNCTION_NAME "${function_name}")

  # Replace template syntax to produce valid token.
  string(REGEX REPLACE "<" "_" FUNCTION_TOKEN ${FUNCTION_NAME})
  string(REGEX REPLACE ">" "" FUNCTION_TOKEN ${FUNCTION_TOKEN})
  string(REGEX REPLACE "," "_" FUNCTION_TOKEN ${FUNCTION_TOKEN})
  string(REGEX REPLACE " " "" FUNCTION_TOKEN ${FUNCTION_TOKEN})

  # Generate test file
  set(out_file
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${FUNCTION_TOKEN}_test.${OUT_FILE_EXT}"
  )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply_test.cpp.in"
    "${out_file}"
  )

  # Add test
  add_executable(${FUNCTION_TOKEN}_test "${out_file}")
  target_include_directories(${FUNCTION_TOKEN}_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  string(REGEX REPLACE "<.*>" "" FUNCTION_LIB_NAME ${FUNCTION_NAME})
  target_link_libraries(${FUNCTION_TOKEN}_test PRIVATE
    GTest::GTest
    GTest::Main
    "${FUNCTION_LIB_NAME}"
  )
  add_test(NAME ${FUNCTION_TOKEN}_test COMMAND ${FUNCTION_TOKEN}_test)
endfunction()

get_property(local_cxx_multipliers GLOBAL PROPERTY
  CXX_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cxx_multipliers)
  generate_multiply_test_for_function(HOST ${multiplier}_test ${multiplier})
endforeach()

get_property(local_cuda_host_multipliers GLOBAL PROPERTY
  CUDA_HOST_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_host_multipliers)
  generate_multiply_test_for_function(HOST ${multiplier}_test "${multiplier}<32>")
endforeach()

get_property(local_cuda_device_multipliers GLOBAL PROPERTY
  CUDA_DEVICE_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_device_multipliers)
  generate_multiply_test_for_function(DEVICE ${multiplier}_test "${multiplier}<32>")
endforeach()

add_executable(SingleThreadedExecutor_test SingleThreadedExecutor_test.cpp)
target_link_libraries(SingleThreadedExecutor_test PRIVATE
  GTest::GTest
  GTest::Main
)
add_test(NAME SingleThreadedExecutor_test COMMAND SingleThreadedExecutor_test)

add_executable(ThreadPoolExecutor_test ThreadPoolExecutor_test.cpp)
target_link_libraries(ThreadPoolExecutor_test PRIVATE
  GTest::GTest
  GTest::Main
)
add_test(NAME ThreadPoolExecutor_test COMMAND ThreadPoolExecutor_test)

add_executable(CudaExecutor_test CudaExecutor_test.cu)
target_link_libraries(CudaExecutor_test PRIVATE
  GTest::GTest
  GTest::Main
)
add_test(NAME CudaExecutor_test COMMAND CudaExecutor_test)

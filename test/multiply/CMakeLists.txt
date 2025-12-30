add_executable(multiply_test multiply_test.cpp)
target_link_libraries(multiply_test
  GTest::GTest
  GTest::Main
  std_transform_multiply
  par_std_transform_multiply
  cuda_multiply
)
add_test(NAME multiply_test COMMAND multiply_test)

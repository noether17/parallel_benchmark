add_executable(multiply_benchmark)
target_link_libraries(multiply_benchmark PRIVATE
  benchmark::benchmark
  benchmark::benchmark_main
)

add_library(warmup_binary_benchmark OBJECT warmup_binary_benchmark.cpp)
target_link_libraries(warmup_binary_benchmark PRIVATE
  benchmark::benchmark
)
target_sources(multiply_benchmark PRIVATE
  $<TARGET_OBJECTS:warmup_binary_benchmark>
)

function(generate_multiply_benchmark_for_function
         benchmarker_type function_name)
  # Configured text
  if(benchmarker_type STREQUAL "HOST")
    set(BENCHMARKER_HEADER MultiplyBenchmarker.hpp)
    set(BENCHMARKER_TEMPLATE MultiplyBenchmarker)
    set(OUT_FILE_EXT cpp)
  elseif(benchmarker_type STREQUAL "DEVICE")
    set(BENCHMARKER_HEADER DeviceMultiplyBenchmarker.cuh)
    set(BENCHMARKER_TEMPLATE DeviceMultiplyBenchmarker)
    set(OUT_FILE_EXT cu)
  else()
    message(FATAL_ERROR "UNKNOWN BENCHMARKER TYPE ${benchmarker_type}!")
  endif()
  set(FUNCTION_NAME "${function_name}")

  # Replace template syntax to produce valid token.
  string(REGEX REPLACE "<" "_" FUNCTION_TOKEN ${FUNCTION_NAME})
  string(REGEX REPLACE ">" "" FUNCTION_TOKEN ${FUNCTION_TOKEN})
  string(REGEX REPLACE "," "_" FUNCTION_TOKEN ${FUNCTION_TOKEN})
  string(REGEX REPLACE " " "" FUNCTION_TOKEN ${FUNCTION_TOKEN})

  # Generate test file
  set(out_file
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${FUNCTION_TOKEN}_benchmark.${OUT_FILE_EXT}"
  )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply_benchmark.cpp.in"
    "${out_file}"
  )

  # Add benchmark
  add_library(${FUNCTION_TOKEN}_benchmark OBJECT "${out_file}")
  target_include_directories(${FUNCTION_TOKEN}_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  string(REGEX REPLACE "<.*>" "" FUNCTION_LIB_NAME ${FUNCTION_NAME})
  target_link_libraries(${FUNCTION_TOKEN}_benchmark PRIVATE
    benchmark::benchmark
    "${FUNCTION_LIB_NAME}"
  )
  target_link_libraries(multiply_benchmark PRIVATE
    "${FUNCTION_LIB_NAME}"
  )
  target_sources(multiply_benchmark PRIVATE
    $<TARGET_OBJECTS:${FUNCTION_TOKEN}_benchmark>
  )
endfunction()

get_property(local_cxx_multipliers GLOBAL PROPERTY
  CXX_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cxx_multipliers)
  generate_multiply_benchmark_for_function(HOST ${multiplier})
endforeach()

get_property(local_cuda_host_multipliers GLOBAL PROPERTY
  CUDA_HOST_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_host_multipliers)
  generate_multiply_benchmark_for_function(HOST "${multiplier}<32>")
  generate_multiply_benchmark_for_function(HOST "${multiplier}<64>")
  generate_multiply_benchmark_for_function(HOST "${multiplier}<128>")
  generate_multiply_benchmark_for_function(HOST "${multiplier}<256>")
  generate_multiply_benchmark_for_function(HOST "${multiplier}<512>")
  generate_multiply_benchmark_for_function(HOST "${multiplier}<1024>")
endforeach()

get_property(local_cuda_device_multipliers GLOBAL PROPERTY
  CUDA_DEVICE_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_device_multipliers)
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<32>")
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<64>")
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<128>")
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<256>")
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<512>")
  generate_multiply_benchmark_for_function(DEVICE "${multiplier}<1024>")
endforeach()

#########
# ParExec
#########
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")

function(generate_par_exec_benchmark PAR_EXEC_TYPE)
  get_par_exec_properties("${PAR_EXEC_TYPE}")
  set(MULTIPLIER "${PAR_EXEC_TOKEN}_multiply")

  if(PAR_EXEC_BASENAME MATCHES "Cuda")
    set(BENCHMARK_TYPE "PAR_EXEC_CUDA_BENCHMARK")
  else()
    set(BENCHMARK_TYPE "PAR_EXEC_CXX_BENCHMARK")
  endif()
  set(BENCHMARK_NAME "${MULTIPLIER}_benchmark")
  message(STATUS "BENCHMARK_NAME: ${BENCHMARK_NAME}")
  set(BENCHMARK_FILE_NAME "${BENCHMARK_NAME}.${PAR_EXEC_SOURCE_EXT}")
  set(BENCHMARK_FILE_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${BENCHMARK_FILE_NAME}"
  )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/par_exec_multiply_benchmark.cpp.in"
    "${BENCHMARK_FILE_PATH}"
  )

  add_library("${BENCHMARK_NAME}" OBJECT "${BENCHMARK_FILE_PATH}")
  target_include_directories("${BENCHMARK_NAME}" PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries("${BENCHMARK_NAME}" PRIVATE
    benchmark::benchmark
    ${MULTIPLIER}
  )
  target_link_libraries(multiply_benchmark PRIVATE
    ${MULTIPLIER}
  )
  target_sources(multiply_benchmark PRIVATE
    $<TARGET_OBJECTS:${BENCHMARK_NAME}>
  )
endfunction()

get_property(TEMP_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PAR_EXEC IN LISTS TEMP_PARALLEL_EXECUTORS)
  generate_par_exec_benchmark(${PAR_EXEC})
endforeach()

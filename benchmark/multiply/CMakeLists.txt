add_executable(multiply_benchmark)
target_link_libraries(multiply_benchmark PRIVATE
  benchmark::benchmark
  benchmark::benchmark_main
)

function(generate_multiplier_benchmark MULTIPLIER)
  message(STATUS "MULTIPLIER: ${MULTIPLIER}")
  if ("${MULTIPLIER}" MATCHES "cuda.*")
    set(BENCHMARK_TYPE "PAR_EXEC_CUDA_BENCHMARK") # TODO: Replace this name.
    set(SOURCE_FILE_EXT "cu")
  else()
    set(BENCHMARK_TYPE "CXX_BENCHMARK")
    set(SOURCE_FILE_EXT "cpp")
  endif()
  message(STATUS "SOURCE_FILE_EXT: ${SOURCE_FILE_EXT}")
  set(MULTIPLIER_HEADER "${MULTIPLIER}.hpp")
  set(BENCHMARK_NAME "${MULTIPLIER}_benchmark")
  set(BENCHMARK_FILE_NAME "${BENCHMARK_NAME}.${SOURCE_FILE_EXT}")
  set(BENCHMARK_FILE_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${BENCHMARK_FILE_NAME}"
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/par_exec_multiply_benchmark.cpp.in"
    "${BENCHMARK_FILE_PATH}"
  )

  # Create the benchmark.
  add_library("${BENCHMARK_NAME}" OBJECT "${BENCHMARK_FILE_PATH}")
  target_include_directories("${BENCHMARK_NAME}" PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries("${BENCHMARK_NAME}" PRIVATE
    benchmark::benchmark
    ${MULTIPLIER}
  )

  # Link to benchmark main file.
  target_link_libraries(multiply_benchmark PRIVATE
    ${MULTIPLIER}
  )
  target_sources(multiply_benchmark PRIVATE
    $<TARGET_OBJECTS:${BENCHMARK_NAME}>
  )
endfunction()

generate_multiplier_benchmark("warmup_binary")

get_property(local_cxx_multipliers GLOBAL PROPERTY CXX_MULTIPLIERS)
foreach(multiplier IN LISTS local_cxx_multipliers)
  generate_multiplier_benchmark("${multiplier}")
endforeach()

get_property(TEMP_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
foreach(CUDA_MULTIPLIER IN LISTS TEMP_CUDA_MULTIPLIERS)
  generate_multiplier_benchmark("${CUDA_MULTIPLIER}")
endforeach()

#########
# ParExec
#########
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")

function(generate_par_exec_benchmark PAR_EXEC_TYPE)
  get_par_exec_properties("${PAR_EXEC_TYPE}")

  # Set multiplier and benchmark name.
  set(MULTIPLIER "${PAR_EXEC_TOKEN}_multiply")
  set(BENCHMARK_NAME "${MULTIPLIER}_benchmark")

  # Configure benchmark file.
  set(MULTIPLIER_HEADER "${MULTIPLIER}.${PAR_EXEC_HEADER_EXT}")
  if(PAR_EXEC_BASENAME MATCHES "Cuda")
    set(BENCHMARK_TYPE "PAR_EXEC_CUDA_BENCHMARK")
  else()
    set(BENCHMARK_TYPE "PAR_EXEC_CXX_BENCHMARK")
  endif()
  set(BENCHMARK_FILE_NAME "${BENCHMARK_NAME}.${PAR_EXEC_SOURCE_EXT}")
  set(BENCHMARK_FILE_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${BENCHMARK_FILE_NAME}"
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/par_exec_multiply_benchmark.cpp.in"
    "${BENCHMARK_FILE_PATH}"
  )

  # Create the benchmark.
  add_library("${BENCHMARK_NAME}" OBJECT "${BENCHMARK_FILE_PATH}")
  target_include_directories("${BENCHMARK_NAME}" PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries("${BENCHMARK_NAME}" PRIVATE
    benchmark::benchmark
    ${MULTIPLIER}
  )

  # Link to benchmark main file.
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

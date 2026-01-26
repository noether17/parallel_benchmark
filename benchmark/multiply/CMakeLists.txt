add_executable(multiply_benchmark)
target_link_libraries(multiply_benchmark PRIVATE
  benchmark::benchmark
  benchmark::benchmark_main
)

function(generate_multiplier_benchmark MULTIPLIER)
  # Get multiplier info.
  set(MULTIPLIER_HEADER "${MULTIPLIER}.hpp")
  get_target_property(SOURCE_FILE_EXT "${MULTIPLIER}" SOURCE_FILE_EXT)
  get_target_property(MULTIPLIER_TYPE "${MULTIPLIER}" MULTIPLIER_TYPE)

  # Set benchmark info.
  set(BENCHMARK_NAME "${MULTIPLIER}_benchmark")
  set(BENCHMARK_FILE_NAME "${BENCHMARK_NAME}.${SOURCE_FILE_EXT}")
  set(BENCHMARK_FILE_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${BENCHMARK_FILE_NAME}"
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/multiply_benchmark.cpp.in"
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

get_property(LOCAL_MULTIPLIERS GLOBAL PROPERTY MULTIPLIERS)
foreach(MULTIPLIER IN LISTS LOCAL_MULTIPLIERS)
  generate_multiplier_benchmark("${MULTIPLIER}")
endforeach()

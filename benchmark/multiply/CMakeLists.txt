set(MULTIPLY_BENCHMARKS
  warmup_binary_benchmark
  single_threaded_multiply_benchmark
  std_transform_multiply_benchmark
  par_std_transform_multiply_benchmark
  cuda_multiply_benchmark
)

foreach(bm IN LISTS MULTIPLY_BENCHMARKS)
  add_library(${bm} OBJECT ${bm}.cpp)
  target_link_libraries(${bm} PRIVATE benchmark::benchmark)
endforeach()

target_link_libraries(single_threaded_multiply_benchmark PRIVATE
  single_threaded_multiply
)
target_link_libraries(std_transform_multiply_benchmark PRIVATE
  std_transform_multiply
)
target_link_libraries(par_std_transform_multiply_benchmark PRIVATE
  par_std_transform_multiply
)
target_link_libraries(cuda_multiply_benchmark PRIVATE
  cuda_multiply
)

add_library(cuda_device_multiply_benchmark OBJECT
  cuda_device_multiply_benchmark.cu
)
target_link_libraries(cuda_device_multiply_benchmark PRIVATE
  benchmark::benchmark
  cuda_multiply
)

add_executable(multiply_benchmark multiply_benchmark.cpp)
target_link_libraries(multiply_benchmark PRIVATE
  benchmark::benchmark
  benchmark::benchmark_main
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
  cuda_multiply
)

foreach(bm IN LISTS MULTIPLY_BENCHMARKS)
  target_sources(multiply_benchmark PRIVATE $<TARGET_OBJECTS:${bm}>)
endforeach()
target_sources(multiply_benchmark PRIVATE
  $<TARGET_OBJECTS:cuda_device_multiply_benchmark>
)

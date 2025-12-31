set(MULTIPLY_BENCHMARKS
  warmup_binary_benchmark
  noop_multiply_benchmark
  single_threaded_multiply_benchmark
  std_transform_multiply_benchmark
  par_std_transform_multiply_benchmark
  cuda_multiply_benchmark
)

foreach(bm IN LISTS MULTIPLY_BENCHMARKS)
  add_library(${bm} OBJECT ${bm}.cpp)
  target_link_libraries(${bm} PRIVATE benchmark::benchmark)
endforeach()

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

#add_library(warmup_binary_benchmark OBJECT
#  warmup_binary_benchmark.cpp
#)
#target_link_libraries(warmup_binary_benchmark PRIVATE
#  benchmark::benchmark
#)
#
#add_library(noop_multiply_benchmark OBJECT
#  noop_multiply_benchmark.cpp
#)
#target_link_libraries(noop_multiply_benchmark PRIVATE
#  benchmark::benchmark
#)
#
#add_library(single_threaded_multiply_benchmark OBJECT
#  single_threaded_multiply_benchmark.cpp
#)
#target_link_libraries(single_threaded_multiply_benchmark PRIVATE
#  benchmark::benchmark
#)
#
#add_library(std_transform_multiply_benchmark OBJECT
#  std_transform_multiply_benchmark.cpp
#)
#target_link_libraries(std_transform_multiply_benchmark PRIVATE
#  benchmark::benchmark
#  std_transform_multiply
#)
#
#add_library(par_std_transform_multiply_benchmark OBJECT
#  par_std_transform_multiply_benchmark.cpp
#)
#target_link_libraries(par_std_transform_multiply_benchmark PRIVATE
#  benchmark::benchmark
#  par_std_transform_multiply
#)

#add_library(cuda_host_multiply_benchmark OBJECT
#  cuda_host_multiply_benchmark.cpp
#)
#target_link_libraries(cuda_host_multiply_benchmark PRIVATE
#  cuda_multiply
#)
#
#add_library(cuda_device_multiply_benchmark OBJECT
#  cuda_device_multiply_benchmark.cu
#)
#target_link_libraries(cuda_device_multiply_benchmark PRIVATE
#  cuda_multiply
#)

#add_executable(multiply_benchmark multiply_benchmark.cpp)
#target_link_libraries(multiply_benchmark PRIVATE
#  benchmark::benchmark
#  benchmark::benchmark_main
#  warmup_binary_benchmark
#  noop_multiply_benchmark
#  single_threaded_multiply_benchmark
#  std_transform_multiply_benchmark
#  par_std_transform_multiply_benchmark
#  #cuda_host_multiply_benchmark
#  #cuda_device_multiply_benchmark
#)
#target_sources(multiply_benchmark PRIVATE
#  $<TARGET_OBJECTS:warmup_binary_benchmark>
#  $<TARGET_OBJECTS:noop_multiply_benchmark>
#  $<TARGET_OBJECTS:single_threaded_multiply_benchmark>
#  $<TARGET_OBJECTS:std_transform_multiply>
#  $<TARGET_OBJECTS:std_transform_multiply_benchmark>
#  $<TARGET_OBJECTS:par_std_transform_multiply>
#  $<TARGET_OBJECTS:par_std_transform_multiply_benchmark>
#  #$<TARGET_OBJECTS:cuda_host_multiply_benchmark>
#  #$<TARGET_OBJECTS:cuda_device_multiply_benchmark>
#)

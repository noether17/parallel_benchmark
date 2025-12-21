find_package(TBB REQUIRED)

add_library(multiply_benchmark OBJECT multiply_benchmark.cpp)
target_link_libraries(multiply_benchmark PRIVATE
  benchmark::benchmark
  TBB::tbb
)

add_executable(parallel_benchmark parallel_benchmark.cpp)
target_link_libraries(parallel_benchmark PRIVATE
  benchmark::benchmark
  multiply_benchmark
)
target_sources(parallel_benchmark PRIVATE
  $<TARGET_OBJECTS:multiply_benchmark>
)

#add_executable(parallel_benchmark parallel_benchmark.cu)
#target_link_libraries(parallel_benchmark
#  benchmark::benchmark
#  benchmark::benchmark_main
#  TBB::tbb
#)

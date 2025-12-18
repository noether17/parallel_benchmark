find_package(TBB REQUIRED)
add_executable(parallel_benchmark parallel_benchmark.cu)
target_link_libraries(parallel_benchmark
  benchmark::benchmark
  benchmark::benchmark_main
  TBB::tbb
)

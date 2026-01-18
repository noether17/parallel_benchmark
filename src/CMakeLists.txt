#find_package(TBB REQUIRED)
#
#add_library(multiply_benchmark OBJECT multiply_benchmark.cpp)
#target_link_libraries(multiply_benchmark PRIVATE
#  benchmark::benchmark
#  TBB::tbb
#)
#
#add_executable(parallel_benchmark parallel_benchmark.cpp)
#target_link_libraries(parallel_benchmark PRIVATE
#  benchmark::benchmark
#  multiply_benchmark
#)
#target_sources(parallel_benchmark PRIVATE
#  $<TARGET_OBJECTS:multiply_benchmark>
#)

#add_executable(parallel_benchmark parallel_benchmark.cu)
#target_link_libraries(parallel_benchmark
#  benchmark::benchmark
#  benchmark::benchmark_main
#  TBB::tbb
#)

# Define PARALLEL_EXECUTORS
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "SingleThreadedExecutor"
)
set(THREADPOOL_NUM_THREADS 1 2 4 8 16 32)
foreach(NUM_THREADS IN LISTS THREADPOOL_NUM_THREADS)
  set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
    "ThreadPoolExecutorTemplate<${NUM_THREADS}>"
  )
endforeach()
set(CUDA_THREADS_PER_BLOCK 32 64 128 256 512 1024)
foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
  set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
    "CudaExecutor<${THREADS_PER_BLOCK}>"
  )
endforeach()
get_property(local_parallel_executors GLOBAL PROPERTY PARALLEL_EXECUTORS)
message(STATUS "PARALLEL_EXECUTORS: ${local_parallel_executors}")

add_subdirectory(multiply)

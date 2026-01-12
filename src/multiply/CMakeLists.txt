# C++ multiply libraries
set_property(GLOBAL PROPERTY CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
get_property(local_cxx_multipliers GLOBAL PROPERTY
  CXX_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cxx_multipliers)
  add_library(${multiplier} OBJECT ${multiplier}.cpp)
endforeach()

find_package(TBB REQUIRED)
target_link_libraries(par_std_transform_multiply PRIVATE
  TBB::tbb
)

# CUDA multiply libraries operating on host data
set_property(GLOBAL PROPERTY CUDA_HOST_MULTIPLIERS
  cuda_multiply
)
get_property(local_cuda_host_multipliers GLOBAL PROPERTY
  CUDA_HOST_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_host_multipliers)
  add_library(${multiplier} OBJECT ${multiplier}.cu)
endforeach()

# CUDA multiply libraries operating on device data
set_property(GLOBAL PROPERTY CUDA_DEVICE_MULTIPLIERS
  cuda_device_multiply
)
get_property(local_cuda_device_multipliers GLOBAL PROPERTY
  CUDA_DEVICE_MULTIPLIERS
)
foreach(multiplier IN LISTS local_cuda_device_multipliers)
  add_library(${multiplier} OBJECT ${multiplier}.cu)
endforeach()

########
#ParExec
########
add_library(SingleThreadedExecutor_multiply OBJECT
  SingleThreadedExecutor_multiply.cpp
)
set(local_parexec_cxx_multipliers
  SingleThreadedExecutor_multiply
)

set(local_threadpool_n_threads 1 2 4 8 16 32)
foreach(N_THREADS IN LISTS local_threadpool_n_threads)
  set(library_name "ThreadPoolExecutor${N_THREADS}_multiply")
  set(out_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${library_name}.cpp")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ThreadPoolExecutor_multiply.cpp.in"
    "${out_file}"
  )
  add_library(${library_name} OBJECT ${out_file})
  set(local_parexec_cxx_multipliers "${local_parexec_cxx_multipliers} \
    ThreadPoolExecutor${N_THREADS}_multiply"
  )
endforeach()
set_property(GLOBAL PROPERTY PAREXEC_CXX_MULTIPLIERS
  local_parexec_cxx_multipliers
)
set_property(GLOBAL PROPERTY PAREXEC_THREADPOOL_N_THREADS
  local_threadpool_n_threads
)

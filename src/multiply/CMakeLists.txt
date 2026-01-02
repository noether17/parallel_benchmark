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

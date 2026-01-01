# C++ multiply libraries
set(CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
foreach(multiplier IN LISTS CXX_MULTIPLIERS)
  add_library(${multiplier} OBJECT ${multiplier}.cpp)
endforeach()

find_package(TBB REQUIRED)
target_link_libraries(par_std_transform_multiply PRIVATE
  TBB::tbb
)

# CUDA multiply libraries operating on host data
set(CUDA_HOST_MULTIPLIERS
  cuda_multiply
)
foreach(multiplier IN LISTS CUDA_HOST_MULTIPLIERS)
  add_library(${multiplier} OBJECT ${multiplier}.cu)
endforeach()

# CUDA multiply libraries operating on device data
set(CUDA_DEVICE_MULTIPLIERS
  cuda_device_multiply
)
foreach(multiplier IN LISTS CUDA_DEVICE_MULTIPLIERS)
  add_library(${multiplier} OBJECT ${multiplier}.cu)
endforeach()

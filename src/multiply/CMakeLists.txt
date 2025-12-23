add_library(std_transform_multiply OBJECT std_transform_multiply.cpp)

find_package(TBB REQUIRED)
add_library(par_std_transform_multiply OBJECT par_std_transform_multiply.cpp)
target_link_libraries(par_std_transform_multiply PRIVATE
  TBB::tbb
)

add_library(cuda_multiply OBJECT cuda_multiply.cu)

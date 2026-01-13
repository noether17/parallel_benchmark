cmake_minimum_required(VERSION 4.0.0)
project(parallel_benchmark LANGUAGES CXX CUDA)

find_package(benchmark 1.8.2 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -O3 \
  -save-temps \
  -masm=intel"
)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES native)

# prevent use of rsp files, which are incompatible with clangd
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
  -Xcompiler -Wall \
  -Xcompiler -Wextra \
  -Xcompiler -Werror \
  -Xcompiler -O3 \
  -Xcompiler -save-temps \
  -Xcompiler -masm=intel \
  --expt-relaxed-constexpr \
  --extended-lambda"
)

enable_testing()
find_package(GTest 1.14.0 REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(include)
include_directories(include/ParallelExecutor)
add_subdirectory(src)
add_subdirectory(benchmark)
add_subdirectory(test)

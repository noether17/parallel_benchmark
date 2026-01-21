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

foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
  set(CUDA_MULTIPLIER "cuda_${THREADS_PER_BLOCK}_multiply")
  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${CUDA_MULTIPLIER}.hpp")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_multiply.hpp.in"
    "${GEN_FILE_DIR}/${CUDA_MULTIPLIER}.hpp"
  )
  set(SOURCE_FILE_PATH "${GEN_FILE_DIR}/${CUDA_MULTIPLIER}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_multiply.cu.in"
    "${GEN_FILE_DIR}/${CUDA_MULTIPLIER}.cu"
  )
  add_library("${CUDA_MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
  target_include_directories("${CUDA_MULTIPLIER}" INTERFACE "${GEN_FILE_DIR}")
  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${CUDA_MULTIPLIER}")
endforeach()
get_property(TEMP_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
message(STATUS "TEMP_CUDA_MULTIPLIERS: ${TEMP_CUDA_MULTIPLIERS}")

########
#ParExec
########
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")

function(generate_par_exec_multipliers PAR_EXEC_TYPE)
  get_par_exec_properties("${PAR_EXEC_TYPE}")

  # Generate header file.
  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(LIB_NAME "${PAR_EXEC_TOKEN}_multiply")
  set(HEADER_FILE_NAME "${LIB_NAME}.${PAR_EXEC_HEADER_EXT}")
  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_NAME "${LIB_NAME}.${PAR_EXEC_SOURCE_EXT}")
  set(SOURCE_FILE_PATH "${GEN_FILE_DIR}/${SOURCE_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.cpp.in"
    "${SOURCE_FILE_PATH}"
  )

  # Create library.
  add_library(${LIB_NAME} OBJECT ${SOURCE_FILE_PATH})
  target_include_directories(${LIB_NAME} INTERFACE ${GEN_FILE_DIR})
endfunction()

get_property(TEMP_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PARALLEL_EXECUTOR IN LISTS TEMP_PARALLEL_EXECUTORS)
  generate_par_exec_multipliers("${PARALLEL_EXECUTOR}")
endforeach()

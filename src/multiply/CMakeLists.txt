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
function(generate_par_exec_multipliers par_exec_type)
  if(${par_exec_type} MATCHES ".*Cuda.*")
    set(HEADER_FILE_EXT cuh)
    set(SOURCE_FILE_EXT cu)
  else()
    set(HEADER_FILE_EXT hpp)
    set(SOURCE_FILE_EXT cpp)
  endif()
  set(PAR_EXEC_TYPE ${par_exec_type})

  # Replace template syntax to produce a valid token.
  string(REGEX REPLACE "<" "_" PAR_EXEC_TOKEN ${PAR_EXEC_TYPE})
  string(REGEX REPLACE ">" "" PAR_EXEC_TOKEN ${PAR_EXEC_TOKEN})
  string(REGEX REPLACE "," "_" PAR_EXEC_TOKEN ${PAR_EXEC_TOKEN})
  string(REGEX REPLACE " " "" PAR_EXEC_TOKEN ${PAR_EXEC_TOKEN})

  # Remove template parameters to get basename
  string(REGEX REPLACE "<.*>" "" PAR_EXEC_BASENAME ${PAR_EXEC_TYPE})
  set(PAR_EXEC_HEADER "${PAR_EXEC_BASENAME}.${HEADER_FILE_EXT}")

  # Generate header file.
  set(SOURCE_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(LIB_NAME "${PAR_EXEC_TOKEN}_multiply")
  set(HEADER_FILE_NAME "${LIB_NAME}.${HEADER_FILE_EXT}")
  set(HEADER_FILE_PATH "${SOURCE_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_NAME "${LIB_NAME}.${SOURCE_FILE_EXT}")
  set(SOURCE_FILE_PATH "${SOURCE_FILE_DIR}/${SOURCE_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.cpp.in"
    "${SOURCE_FILE_PATH}"
  )

  # Create library.
  add_library(${LIB_NAME} OBJECT ${SOURCE_FILE_PATH})
  target_include_directories(${LIB_NAME} INTERFACE ${SOURCE_FILE_DIR})
  set_property(GLOBAL APPEND PROPERTY PAR_EXEC_MULTIPLIERS
    ${LIB_NAME}
  )
endfunction()

get_property(TEMP_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PARALLEL_EXECUTOR IN LISTS TEMP_PARALLEL_EXECUTORS)
  message(STATUS "PARALLEL_EXECUTOR: ${PARALLEL_EXECUTOR}")
  generate_par_exec_multipliers("${PARALLEL_EXECUTOR}")
endforeach()
get_property(temp GLOBAL PROPERTY PAR_EXEC_MULTIPLIERS)
message(STATUS "PAR_EXEC_MULTIPLIERS: ${temp}")

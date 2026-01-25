function(generate_multiplier_header MULTIPLIER)
  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply.hpp.in"
    "${GEN_FILE_DIR}/${MULTIPLIER}.hpp"
  )
endfunction()

# C++ multiply libraries
set_property(GLOBAL PROPERTY CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
get_property(TEMP_CXX_MULTIPLIERS GLOBAL PROPERTY
  CXX_MULTIPLIERS
)
foreach(MULTIPLIER IN LISTS TEMP_CXX_MULTIPLIERS)
  generate_multiplier_header(${MULTIPLIER})
  add_library(${MULTIPLIER} OBJECT ${MULTIPLIER}.cpp)
  target_include_directories(${MULTIPLIER} PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )
endforeach()

find_package(TBB REQUIRED)
target_link_libraries(par_std_transform_multiply PRIVATE
  TBB::tbb
)

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
  target_include_directories("${CUDA_MULTIPLIER}" PUBLIC "${GEN_FILE_DIR}")
  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${CUDA_MULTIPLIER}")
endforeach()
get_property(TEMP_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
message(STATUS "TEMP_CUDA_MULTIPLIERS: ${TEMP_CUDA_MULTIPLIERS}")

#########
# ParExec
#########
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

#######################
# Cuda Host Multipliers
#######################
foreach(CUDA_MULTIPLIER IN LISTS TEMP_CUDA_MULTIPLIERS)
  set(HOST_CUDA_MULTIPLIER "host_${CUDA_MULTIPLIER}")
  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(HEADER_FILE_NAME "${HOST_CUDA_MULTIPLIER}.cuh")
  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_multiply.cuh.in"
    "${HEADER_FILE_PATH}"
  )

  add_library("${HOST_CUDA_MULTIPLIER}" OBJECT "${HEADER_FILE_PATH}")
  set_target_properties("${HOST_CUDA_MULTIPLIER}" PROPERTIES LINKER_LANGUAGE
    CUDA
  )
  set_property(GLOBAL APPEND PROPERTY HOST_CUDA_MULTIPLIERS
    "${HOST_CUDA_MULTIPLIER}"
  )
endforeach()
get_property(TEMP_HOST_CUDA_MULTIPLIERS GLOBAL PROPERTY HOST_CUDA_MULTIPLIERS)
message(STATUS "TEMP_HOST_CUDA_MULTIPLIERS: ${TEMP_HOST_CUDA_MULTIPLIERS}")

set(GENERATED_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

#############################
# Create multiplier libraries
#############################

# Standalone C++ libraries
set_property(GLOBAL PROPERTY CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
get_property(LOCAL_CXX_MULTIPLIERS GLOBAL PROPERTY CXX_MULTIPLIERS)
foreach(MULTIPLIER IN LISTS LOCAL_CXX_MULTIPLIERS)
  message(STATUS "Creating library for ${MULTIPLIER}.")
  # Generate header file.
  set(HEADER_FILE_NAME "${MULTIPLIER}.hpp")
  set(HEADER_FILE_PATH "${GENERATED_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Set source file.
  set(SOURCE_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${MULTIPLIER}.cpp")

  # Create the library.
  add_library("${MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
  target_include_directories("${MULTIPLIER}" PUBLIC "${GENERATED_FILE_DIR}")
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${HEADER_FILE_NAME}"
  )
  set_property(GLOBAL APPEND PROPERTY MULTIPLIERS "${MULTIPLIER}")
endforeach()

# Special package needed for parallel std::transform.
# TODO: Make this optional and turn off compilation of
#       par_std_transform_multiply if not present.
find_package(TBB REQUIRED)
target_link_libraries(par_std_transform_multiply PRIVATE TBB::tbb)

# Generate cuda multipliers with varying threads per block.
foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
  set(MULTIPLIER "cuda_${THREADS_PER_BLOCK}_multiply")
  message(STATUS "Creating library for ${MULTIPLIER}.")

  # Generate header file.
  set(HEADER_FILE_NAME "${MULTIPLIER}.hpp")
  set(HEADER_FILE_PATH "${GENERATED_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_PATH "${GENERATED_FILE_DIR}/${MULTIPLIER}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_multiply.cu.in"
    "${SOURCE_FILE_PATH}"
  )

  # Create the library.
  add_library("${MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
  target_include_directories("${MULTIPLIER}" PUBLIC "${GENERATED_FILE_DIR}")
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${HEADER_FILE_NAME}"
  )
  set_property(GLOBAL APPEND PROPERTY MULTIPLIERS "${MULTIPLIER}")
  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
endforeach()

# Generate ParExec multipliers.
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")
get_property(LOCAL_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PARALLEL_EXECUTOR IN LISTS LOCAL_PARALLEL_EXECUTORS)
  get_par_exec_properties("${PARALLEL_EXECUTOR}")
  set(MULTIPLIER "${PAR_EXEC_TOKEN}_multiply")
  message(STATUS "Creating library for ${MULTIPLIER}.")

  # Generate header file.
  set(HEADER_FILE_NAME "${MULTIPLIER}.${PAR_EXEC_HEADER_EXT}")
  set(HEADER_FILE_PATH
    "${GENERATED_FILE_DIR}/${MULTIPLIER}.${PAR_EXEC_HEADER_EXT}"
  )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_PATH
    "${GENERATED_FILE_DIR}/${MULTIPLIER}.${PAR_EXEC_SOURCE_EXT}"
  )
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.cpp.in"
    "${SOURCE_FILE_PATH}"
  )

  # Create the library.
  add_library("${MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
  target_include_directories("${MULTIPLIER}" PUBLIC "${GENERATED_FILE_DIR}")
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${HEADER_FILE_NAME}"
  )
  set_property(GLOBAL APPEND PROPERTY MULTIPLIERS "${MULTIPLIER}")
  if("${MULTIPLIER}" MATCHES "^CudaExecutor.*")
    set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
  endif()
endforeach()

# Generate cuda multipliers operating on host data.
get_property(LOCAL_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
foreach(CUDA_MULTIPLIER IN LISTS LOCAL_CUDA_MULTIPLIERS)
  set(MULTIPLIER "host_${CUDA_MULTIPLIER}")
  message(STATUS "Creating library for ${MULTIPLIER}.")

  # Generate header file.
  get_target_property(CUDA_MULTIPLIER_HEADER "${CUDA_MULTIPLIER}"
    HEADER_FILE_NAME
  )
  set(HEADER_FILE_PATH "${GENERATED_FILE_DIR}/${MULTIPLIER}.hpp")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_multiply.hpp.in"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_PATH "${GENERATED_FILE_DIR}/${MULTIPLIER}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_multiply.cu.in"
    "${SOURCE_FILE_PATH}"
  )

  # Create the library.
  add_library("${MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
  target_include_directories("${MULTIPLIER}" PUBLIC "${GENERATED_FILE_DIR}")
  target_link_libraries("${MULTIPLIER}" PUBLIC "${CUDA_MULTIPLIER}")
  target_sources("${MULTIPLIER}" PUBLIC $<TARGET_OBJECTS:${CUDA_MULTIPLIER}>)
  set_property(GLOBAL APPEND PROPERTY MULTIPLIERS "${MULTIPLIER}")
endforeach()








#function(generate_multiplier_header MULTIPLIER)
#  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply.hpp.in"
#    "${GEN_FILE_DIR}/${MULTIPLIER}.hpp"
#  )
#endfunction()
#
## C++ multiply libraries
#set_property(GLOBAL PROPERTY CXX_MULTIPLIERS
#  single_threaded_multiply
#  std_transform_multiply
#  par_std_transform_multiply
#)
#get_property(TEMP_CXX_MULTIPLIERS GLOBAL PROPERTY
#  CXX_MULTIPLIERS
#)
#foreach(MULTIPLIER IN LISTS TEMP_CXX_MULTIPLIERS)
#  generate_multiplier_header(${MULTIPLIER})
#  add_library(${MULTIPLIER} OBJECT ${MULTIPLIER}.cpp)
#  target_include_directories(${MULTIPLIER} PUBLIC
#    "${CMAKE_CURRENT_BINARY_DIR}/generated"
#  )
#endforeach()
#
#find_package(TBB REQUIRED)
#target_link_libraries(par_std_transform_multiply PRIVATE
#  TBB::tbb
#)
#
#foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
#  set(MULTIPLIER "cuda_${THREADS_PER_BLOCK}_multiply")
#  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
#  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${MULTIPLIER}.hpp")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/multiply.hpp.in"
#    "${GEN_FILE_DIR}/${MULTIPLIER}.hpp"
#  )
#  set(SOURCE_FILE_PATH "${GEN_FILE_DIR}/${MULTIPLIER}.cu")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_multiply.cu.in"
#    "${GEN_FILE_DIR}/${MULTIPLIER}.cu"
#  )
#  add_library("${MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
#  target_include_directories("${MULTIPLIER}" PUBLIC "${GEN_FILE_DIR}")
#  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
#endforeach()
#get_property(TEMP_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
#message(STATUS "TEMP_CUDA_MULTIPLIERS: ${TEMP_CUDA_MULTIPLIERS}")
#
##########
## ParExec
##########
#include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")
#
#function(generate_par_exec_multipliers PAR_EXEC_TYPE)
#  get_par_exec_properties("${PAR_EXEC_TYPE}")
#
#  # Generate header file.
#  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
#  set(LIB_NAME "${PAR_EXEC_TOKEN}_multiply")
#  set(HEADER_FILE_NAME "${LIB_NAME}.${PAR_EXEC_HEADER_EXT}")
#  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${HEADER_FILE_NAME}")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.hpp.in"
#    "${HEADER_FILE_PATH}"
#  )
#
#  # Generate source file.
#  set(SOURCE_FILE_NAME "${LIB_NAME}.${PAR_EXEC_SOURCE_EXT}")
#  set(SOURCE_FILE_PATH "${GEN_FILE_DIR}/${SOURCE_FILE_NAME}")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParExec_multiply.cpp.in"
#    "${SOURCE_FILE_PATH}"
#  )
#
#  # Create library.
#  add_library(${LIB_NAME} OBJECT ${SOURCE_FILE_PATH})
#  target_include_directories(${LIB_NAME} INTERFACE ${GEN_FILE_DIR})
#endfunction()
#
#get_property(TEMP_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
#foreach(PARALLEL_EXECUTOR IN LISTS TEMP_PARALLEL_EXECUTORS)
#  generate_par_exec_multipliers("${PARALLEL_EXECUTOR}")
#endforeach()
#
########################
## Cuda Host Multipliers
########################
#foreach(CUDA_MULTIPLIER IN LISTS TEMP_CUDA_MULTIPLIERS)
#  set(HOST_CUDA_MULTIPLIER "host_${CUDA_MULTIPLIER}")
#  set(GEN_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
#  set(HEADER_FILE_NAME "${HOST_CUDA_MULTIPLIER}.hpp")
#  set(HEADER_FILE_PATH "${GEN_FILE_DIR}/${HEADER_FILE_NAME}")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_multiply.hpp.in"
#    "${HEADER_FILE_PATH}"
#  )
#
#  set(SOURCE_FILE_NAME "${HOST_CUDA_MULTIPLIER}.cu")
#  set(SOURCE_FILE_PATH "${GEN_FILE_DIR}/${SOURCE_FILE_NAME}")
#  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_multiply.cu.in"
#    "${SOURCE_FILE_PATH}"
#  )
#
#  add_library("${HOST_CUDA_MULTIPLIER}" OBJECT "${SOURCE_FILE_PATH}")
#  target_include_directories("${HOST_CUDA_MULTIPLIER}" PUBLIC "${GEN_FILE_DIR}")
#  target_link_libraries("${HOST_CUDA_MULTIPLIER}" PUBLIC "${CUDA_MULTIPLIER}")
#  target_sources("${HOST_CUDA_MULTIPLIER}" PUBLIC
#    $<TARGET_OBJECTS:${CUDA_MULTIPLIER}>
#  )
#  set_property(GLOBAL APPEND PROPERTY HOST_CUDA_MULTIPLIERS
#    "${HOST_CUDA_MULTIPLIER}"
#  )
#endforeach()
#get_property(TEMP_HOST_CUDA_MULTIPLIERS GLOBAL PROPERTY HOST_CUDA_MULTIPLIERS)
#message(STATUS "TEMP_HOST_CUDA_MULTIPLIERS: ${TEMP_HOST_CUDA_MULTIPLIERS}")

set(GENERATED_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

#############################
# Create multiplier libraries
#############################

function(create_library MULTIPLIER)
  # Required properties.
  get_target_property(HEADER_FILE_NAME "${MULTIPLIER}" HEADER_FILE_NAME)
  get_target_property(HEADER_TEMPLATE_NAME "${MULTIPLIER}" HEADER_TEMPLATE_NAME)
  get_target_property(SOURCE_FILE_NAME "${MULTIPLIER}" SOURCE_FILE_NAME)
  get_target_property(SOURCE_TEMPLATE_NAME "${MULTIPLIER}" SOURCE_TEMPLATE_NAME)

  # Optional properties (used in configuring specific template files).
  # TODO

  # Generate header file.
  set(HEADER_FILE_PATH "${GENERATED_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_TEMPLATE_NAME}"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_PATH "${GENERATED_FILE_DIR}/${SOURCE_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_TEMPLATE_NAME}"
    "${SOURCE_FILE_PATH}"
  )

  # Create the library.
  target_sources("${MULTIPLIER}" PRIVATE "${SOURCE_FILE_PATH}")
  target_include_directories("${MULTIPLIER}" PUBLIC "${GENERATED_FILE_DIR}")
  set_property(GLOBAL APPEND PROPERTY MULTIPLIERS "${MULTIPLIER}")
endfunction()

# Standalone C++ libraries
set_property(GLOBAL PROPERTY CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
get_property(LOCAL_CXX_MULTIPLIERS GLOBAL PROPERTY CXX_MULTIPLIERS)
foreach(MULTIPLIER IN LISTS LOCAL_CXX_MULTIPLIERS)
  message(STATUS "Creating library for ${MULTIPLIER}.")
  add_library("${MULTIPLIER}" STATIC)
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${MULTIPLIER}.hpp"
    HEADER_TEMPLATE_NAME "multiply.hpp.in"
    SOURCE_FILE_NAME "${MULTIPLIER}.cpp"
    SOURCE_TEMPLATE_NAME "${MULTIPLIER}.cpp"
  )
  create_library("${MULTIPLIER}")
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
  add_library("${MULTIPLIER}" STATIC)
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${MULTIPLIER}.hpp"
    HEADER_TEMPLATE_NAME "multiply.hpp.in"
    SOURCE_FILE_NAME "${MULTIPLIER}.cu"
    SOURCE_TEMPLATE_NAME "cuda_multiply.cu.in"
  )
  create_library("${MULTIPLIER}")
  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
endforeach()

# Generate ParExec multipliers.
include("${CMAKE_SOURCE_DIR}/src/get_par_exec_properties.cmake")
get_property(LOCAL_PARALLEL_EXECUTORS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PARALLEL_EXECUTOR IN LISTS LOCAL_PARALLEL_EXECUTORS)
  get_par_exec_properties("${PARALLEL_EXECUTOR}")
  set(MULTIPLIER "${PAR_EXEC_TOKEN}_multiply")
  message(STATUS "Creating library for ${MULTIPLIER}.")
  add_library("${MULTIPLIER}" STATIC)
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${MULTIPLIER}.hpp"
    HEADER_TEMPLATE_NAME "ParExec_multiply.hpp.in"
    SOURCE_FILE_NAME "${MULTIPLIER}.${PAR_EXEC_SOURCE_EXT}"
    SOURCE_TEMPLATE_NAME "ParExec_multiply.cpp.in"
  )
  create_library("${MULTIPLIER}")
  if("${MULTIPLIER}" MATCHES "^CudaExecutor.*")
    set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
  endif()
endforeach()

# Generate cuda multipliers operating on host data.
get_property(LOCAL_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
foreach(CUDA_MULTIPLIER IN LISTS LOCAL_CUDA_MULTIPLIERS)
  set(MULTIPLIER "host_${CUDA_MULTIPLIER}")
  message(STATUS "Creating library for ${MULTIPLIER}.")
  add_library("${MULTIPLIER}" STATIC)
  get_target_property(CUDA_MULTIPLIER_HEADER
    ${CUDA_MULTIPLIER} HEADER_FILE_NAME
  )
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_FILE_NAME "${MULTIPLIER}.hpp"
    HEADER_TEMPLATE_NAME "cuda_host_multiply.hpp.in"
    SOURCE_FILE_NAME "${MULTIPLIER}.cu"
    SOURCE_TEMPLATE_NAME "cuda_host_multiply.cu.in"
  )
  create_library("${MULTIPLIER}")
  target_link_libraries("${MULTIPLIER}" PUBLIC "${CUDA_MULTIPLIER}")
endforeach()

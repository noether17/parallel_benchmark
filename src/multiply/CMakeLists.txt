set(GENERATED_FILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

#############################
# Create multiplier libraries
#############################

function(create_library MULTIPLIER)
  get_target_property(HEADER_TEMPLATE_NAME "${MULTIPLIER}" HEADER_TEMPLATE_NAME)
  get_target_property(SOURCE_TEMPLATE_NAME "${MULTIPLIER}" SOURCE_TEMPLATE_NAME)
  get_target_property(SOURCE_FILE_EXT "${MULTIPLIER}" SOURCE_FILE_EXT)

  # Generate header file.
  set(HEADER_FILE_NAME "${MULTIPLIER}.hpp")
  set(HEADER_FILE_PATH "${GENERATED_FILE_DIR}/${HEADER_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_TEMPLATE_NAME}"
    "${HEADER_FILE_PATH}"
  )

  # Generate source file.
  set(SOURCE_FILE_NAME "${MULTIPLIER}.${SOURCE_FILE_EXT}")
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
set(CXX_MULTIPLIERS
  single_threaded_multiply
  std_transform_multiply
  par_std_transform_multiply
)
foreach(MULTIPLIER IN LISTS CXX_MULTIPLIERS)
  add_library("${MULTIPLIER}" STATIC)
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_TEMPLATE_NAME "multiply.hpp.in"
    SOURCE_TEMPLATE_NAME "${MULTIPLIER}.cpp"
    SOURCE_FILE_EXT "cpp"
    MULTIPLIER_FAMILY "CXX_MULTIPLIER"
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
  add_library("${MULTIPLIER}" STATIC)
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_TEMPLATE_NAME "multiply.hpp.in"
    SOURCE_TEMPLATE_NAME "cuda_multiply.cu.in"
    SOURCE_FILE_EXT "cu"
    MULTIPLIER_FAMILY "CUDA_MULTIPLIER"
  )
  create_library("${MULTIPLIER}")
  set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
endforeach()

# Generate ParD multipliers.
get_property(LOCAL_PARALLEL_DOERS GLOBAL PROPERTY PARALLEL_DOERS)
foreach(PARALLEL_DOER IN LISTS LOCAL_PARALLEL_DOERS)
  set(MULTIPLIER "${PARALLEL_DOER}_multiply")
  add_library("${MULTIPLIER}" STATIC)
  get_target_property(PARD_BASENAME "${PARALLEL_DOER}" BASENAME)
  get_target_property(PARD_FAMILY "${PARALLEL_DOER}" FAMILY)
  get_target_property(PARD_HEADER_FILE "${PARALLEL_DOER}" HEADER_FILE_PATH)
  get_target_property(PARD_SOURCE_EXT "${PARALLEL_DOER}" SOURCE_FILE_EXT)
  get_target_property(PARD_TYPE "${PARALLEL_DOER}" TYPENAME)

  if("${PARD_FAMILY}" STREQUAL "CXX_DOER")
    set(MULTIPLIER_FAMILY "CXX_MULTIPLIER")
  elseif("${PARD_FAMILY}" STREQUAL "CXX_THREADPOOL_DOER")
    # ThreadPool requires special setup.
    set(MULTIPLIER_FAMILY "CXX_THREADPOOL_MULTIPLIER")
  elseif("${PARD_FAMILY}" STREQUAL "CUDA_DOER")
    set(MULTIPLIER_FAMILY "CUDA_MULTIPLIER")
  endif()

  if("${MULTIPLIER_FAMILY}" STREQUAL "CXX_THREADPOOL_MULTIPLIER")
    set(HEADER_TEMPLATE_NAME "ParD_multiply.hpp.in")
  else()
    set(HEADER_TEMPLATE_NAME "multiply.hpp.in")
  endif()

  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_TEMPLATE_NAME "${HEADER_TEMPLATE_NAME}"
    SOURCE_TEMPLATE_NAME "ParD_multiply.cpp.in"
    SOURCE_FILE_EXT "${PARD_SOURCE_EXT}"
    MULTIPLIER_FAMILY "${MULTIPLIER_FAMILY}"
  )
  create_library("${MULTIPLIER}")
  target_link_libraries("${MULTIPLIER}" INTERFACE ParD::ParD)
  if("${MULTIPLIER}" MATCHES "^CudaDoer.*")
    set_property(GLOBAL APPEND PROPERTY CUDA_MULTIPLIERS "${MULTIPLIER}")
  endif()
endforeach()

# Generate cuda multipliers operating on host data.
get_property(LOCAL_CUDA_MULTIPLIERS GLOBAL PROPERTY CUDA_MULTIPLIERS)
foreach(CUDA_MULTIPLIER IN LISTS LOCAL_CUDA_MULTIPLIERS)
  set(MULTIPLIER "host_${CUDA_MULTIPLIER}")
  add_library("${MULTIPLIER}" STATIC)
  set(CUDA_MULTIPLIER_HEADER "${CUDA_MULTIPLIER}.hpp")
  set_target_properties("${MULTIPLIER}" PROPERTIES
    HEADER_TEMPLATE_NAME "cuda_host_multiply.hpp.in"
    SOURCE_TEMPLATE_NAME "cuda_host_multiply.cu.in"
    SOURCE_FILE_EXT "cu"
    MULTIPLIER_FAMILY "HOST_CUDA_MULTIPLIER"
  )
  create_library("${MULTIPLIER}")
  target_link_libraries("${MULTIPLIER}" PUBLIC "${CUDA_MULTIPLIER}")
endforeach()

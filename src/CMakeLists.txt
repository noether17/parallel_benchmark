# Define PARALLEL_DOERS
function(create_parallel_doer BASENAME FAMILY)
  # Set name.
  set(PARALLEL_DOER_NAME "${BASENAME}")
  foreach(TEMPLATE_PARAM IN LISTS ARGN)
    list(APPEND TEMPLATE_PARAMS "${TEMPLATE_PARAM}")
    set(PARALLEL_DOER_NAME "${PARALLEL_DOER_NAME}_${TEMPLATE_PARAM}")
  endforeach()

  # Set typename.
  set(TYPENAME "${BASENAME}")
  if(TEMPLATE_PARAMS)
    list(JOIN TEMPLATE_PARAMS ", " TEMPLATE_PARAMS_STRING)
    set(TYPENAME "${TYPENAME}<${TEMPLATE_PARAMS_STRING}>")
  endif()

  # Get header and source file extensions.
  if("${BASENAME}" MATCHES "^Cuda")
    set(HEADER_FILE_EXT "cuh")
    set(SOURCE_FILE_EXT "cu")
  else()
    set(HEADER_FILE_EXT "hpp")
    set(SOURCE_FILE_EXT "cpp")
  endif()

  # Set header file name for ParD implementation.
  if("${BASENAME}" MATCHES "ThreadPoolTemplateDoer")
    set(HEADER_FILE_PATH "ParD/ThreadPoolDoer.hpp")
  else()
    set(HEADER_FILE_PATH "ParD/${BASENAME}.${HEADER_FILE_EXT}")
  endif()

  # Create target and set properties.
  add_custom_target("${PARALLEL_DOER_NAME}")
  set_target_properties("${PARALLEL_DOER_NAME}" PROPERTIES
    BASENAME "${BASENAME}"
    FAMILY "${FAMILY}"
    HEADER_FILE_PATH "${HEADER_FILE_PATH}"
    SOURCE_FILE_EXT "${SOURCE_FILE_EXT}"
    TYPENAME "${TYPENAME}"
  )
  set_property(GLOBAL APPEND PROPERTY PARALLEL_DOERS
    "${PARALLEL_DOER_NAME}"
  )
endfunction()

create_parallel_doer("SingleThreadedDoer" "CXX_DOER")
set(THREADPOOL_NUM_THREADS 1 2 4 8 16 32)
foreach(NUM_THREADS IN LISTS THREADPOOL_NUM_THREADS)
  create_parallel_doer(
    "ThreadPoolTemplateDoer"
    "CXX_THREADPOOL_DOER"
    "${NUM_THREADS}"
  )
endforeach()
set(CUDA_THREADS_PER_BLOCK 32 64 128 256 512 1024)
foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
  create_parallel_doer("CudaDoer" "CUDA_DOER" "${THREADS_PER_BLOCK}")
endforeach()

add_subdirectory(multiply)

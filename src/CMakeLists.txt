# Define PARALLEL_EXECUTORS
function(create_parallel_executor BASENAME)
  # Set name.
  set(PARALLEL_EXECUTOR_NAME "${BASENAME}")
  foreach(TEMPLATE_PARAM IN LISTS ARGN)
    list(APPEND TEMPLATE_PARAMS "${TEMPLATE_PARAM}")
    set(PARALLEL_EXECUTOR_NAME "${PARALLEL_EXECUTOR_NAME}_${TEMPLATE_PARAM}")
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

  # Set header file name for ParExec implementation.
  set(HEADER_FILE_NAME "${BASENAME}.${HEADER_FILE_EXT}")

  # Create target and set properties.
  add_custom_target("${PARALLEL_EXECUTOR_NAME}")
  set_target_properties("${PARALLEL_EXECUTOR_NAME}" PROPERTIES
    BASENAME "${BASENAME}"
    HEADER_FILE_NAME "${HEADER_FILE_NAME}"
    SOURCE_FILE_EXT "${SOURCE_FILE_EXT}"
    TYPENAME "${TYPENAME}"
  )
  set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
    "${PARALLEL_EXECUTOR_NAME}"
  )
endfunction()

create_parallel_executor("SingleThreadedExecutor")
set(THREADPOOL_NUM_THREADS 1 2 4 8 16 32)
foreach(NUM_THREADS IN LISTS THREADPOOL_NUM_THREADS)
  create_parallel_executor("ThreadPoolExecutorTemplate" "${NUM_THREADS}")
endforeach()
set(CUDA_THREADS_PER_BLOCK 32 64 128 256 512 1024)
foreach(THREADS_PER_BLOCK IN LISTS CUDA_THREADS_PER_BLOCK)
  create_parallel_executor("CudaExecutor" "${THREADS_PER_BLOCK}")
endforeach()

add_subdirectory(multiply)

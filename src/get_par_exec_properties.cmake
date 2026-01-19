include_guard()

macro(get_par_exec_properties PAR_EXEC_TYPE)
  # Get header and source file extensions.
  if("${PAR_EXEC_TYPE}" MATCHES ".*Cuda.*")
    set(PAR_EXEC_HEADER_EXT cuh)
    set(PAR_EXEC_SOURCE_EXT cu)
  else()
    set(PAR_EXEC_HEADER_EXT hpp)
    set(PAR_EXEC_SOURCE_EXT cpp)
  endif()

  # Get basename by removing template parameters.
  string(REGEX REPLACE "<.*>" "" PAR_EXEC_BASENAME "${PAR_EXEC_TYPE}")

  # Get template parameters.
  if("${PAR_EXEC_TYPE}" MATCHES "<.*>")
    string(REGEX REPLACE "[^<]*<" "" PAR_EXEC_TEMPLATE_PARAMS
      "${PAR_EXEC_TYPE}"
    )
    string(REGEX REPLACE ">[^>]*" "" PAR_EXEC_TEMPLATE_PARAMS
      "${PAR_EXEC_TEMPLATE_PARAMS}"
    )
  endif()

  # Get valid token from basename and template parameters.
  set(PAR_EXEC_TOKEN "${PAR_EXEC_BASENAME}")
  if(PAR_EXEC_TEMPLATE_PARAMS)
    set(PAR_EXEC_TOKEN "${PAR_EXEC_TOKEN}_${PAR_EXEC_TEMPLATE_PARAMS}")
    string(REGEX REPLACE "," "_" PAR_EXEC_TOKEN "${PAR_EXEC_TOKEN}")
    string(REGEX REPLACE " " "" PAR_EXEC_TOKEN "${PAR_EXEC_TOKEN}")
  endif()

  # Get ParExec header file name.
  set(PAR_EXEC_HEADER_FILE "${PAR_EXEC_BASENAME}.${PAR_EXEC_HEADER_EXT}")
endmacro()

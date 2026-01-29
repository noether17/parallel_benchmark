add_library(warmup_binary STATIC "warmup_binary.hpp")
set_target_properties(warmup_binary PROPERTIES
  LINKER_LANGUAGE CXX
  SOURCE_FILE_EXT "cpp"
  MULTIPLIER_FAMILY "CXX_MULTIPLIER"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(multiply)

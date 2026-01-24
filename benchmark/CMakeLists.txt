add_library(warmup_binary OBJECT "warmup_binary.hpp")
set_target_properties(warmup_binary PROPERTIES LINKER_LANGUAGE CXX)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(multiply)

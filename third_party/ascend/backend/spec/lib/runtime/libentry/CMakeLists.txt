add_library(entryC SHARED
  libentry.cpp
)
target_compile_options(entryC PRIVATE -fexceptions -frtti)

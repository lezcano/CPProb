file(REMOVE_RECURSE
  "libCPProb_lib.pdb"
  "libCPProb_lib.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/CPProb_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

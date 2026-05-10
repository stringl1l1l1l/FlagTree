add_subdirectory(AutoBlockify)
add_subdirectory(Conversion/TritonToHFusion)
add_subdirectory(Conversion/TritonToHIVM)
add_subdirectory(Conversion/TritonToLLVM)
add_subdirectory(TritonAffinityOpt)

if(TRITON_ENABLE_COVERAGE_HITEST)
  set(_instrument_targets
      DiscreteMaskAccessConversion
      TritonToAnnotation
      TritonToHFusion
      TritonToHIVM
      TritonToLinalg
      TritonToLLVM
      TritonToStructured
      TritonToUnstructure
      MLIRTritonNPUUtils           # from Utils
      TritonAscendIR               # from Dialect/TritonAscend/IR
      TritonStructuredIR           # from Dialect/TritonStructured/IR
      AutoBlockify
      TritonAffinityOpt
  )

  foreach(_target ${_instrument_targets})
      if(TARGET ${_target})
          set_target_properties(${_target} PROPERTIES
              RULE_LAUNCH_COMPILE "hitestwrapper"
              RULE_LAUNCH_LINK "hitestwrapper"
          )
          message(STATUS "Enabled hitestwrapper for target: ${_target}")
      else()
          message(WARNING "Target ${_target} not found, please check the actual target name")
      endif()
  endforeach()
endif()

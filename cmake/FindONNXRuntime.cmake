find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH)

find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()

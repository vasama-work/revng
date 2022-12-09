#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

function(tuple_tree_generator_impl)
  set(oneValueArgs
      TARGET_NAME
      GENERATED_HEADERS_VARIABLE
      GENERATED_IMPLS_VARIABLE
      NAMESPACE
      SCHEMA_PATH
      HEADERS_DIR
      INCLUDE_PATH_PREFIX
      JSONSCHEMA_PATH
      ROOT_TYPE
      GLOBAL_NAME
      PYTHON_PATH
      TYPESCRIPT_PATH)
  set(multiValueArgs HEADERS TYPESCRIPT_INCLUDE STRING_TYPES
                     SEPARATE_STRING_TYPES SCALAR_TYPES)
  cmake_parse_arguments(GENERATOR "" "${oneValueArgs}" "${multiValueArgs}"
                        "${ARGN}")
  if(NOT DEFINED GENERATOR_JSONSCHEMA_PATH)
    set(GENERATOR_JSONSCHEMA_PATH "")
  endif()
  if(NOT DEFINED GENERATOR_ROOT_TYPE)
    set(GENERATOR_ROOT_TYPE "")
  endif()
  if(NOT DEFINED GENERATOR_GLOBAL_NAME)
    set(GENERATOR_GLOBAL_NAME "")
  endif()
  if(NOT DEFINED GENERATOR_STRING_TYPES)
    set(GENERATOR_STRING_TYPES "")
  endif()
  if(NOT DEFINED GENERATOR_SEPARATE_STRING_TYPES)
    set(GENERATOR_SEPARATE_STRING_TYPES "")
  endif()
  if(NOT DEFINED GENERATOR_SCALAR_TYPES)
    set(GENERATOR_SCALAR_TYPES "")
  endif()
  if(NOT DEFINED GENERATOR_PYTHON_PATH)
    set(GENERATOR_PYTHON_PATH "")
  endif()
  if(NOT DEFINED GENERATOR_TYPESCRIPT_PATH)
    set(GENERATOR_TYPESCRIPT_PATH "")
  endif()

  #
  # Collect all the definitions in a single YAML document
  #
  tuple_tree_generator_extract_definitions_from_headers(
    "${GENERATOR_HEADERS}" "${GENERATOR_SCHEMA_PATH}")

  #
  # C++ headers and implementation generation
  #
  tuple_tree_generator_compute_generated_cpp_files(
    "${GENERATOR_HEADERS}" "${GENERATOR_HEADERS_DIR}" LOCAL_GENERATED_HEADERS
    LOCAL_GENERATED_IMPLS)

  tuple_tree_generator_generate_cpp(
    "${GENERATOR_SCHEMA_PATH}"
    "${GENERATOR_NAMESPACE}"
    "${GENERATOR_HEADERS_DIR}"
    "${GENERATOR_INCLUDE_PATH_PREFIX}"
    "${LOCAL_GENERATED_HEADERS}"
    "${LOCAL_GENERATED_IMPLS}"
    "${GENERATOR_ROOT_TYPE}"
    "${GENERATOR_SCALAR_TYPES}")

  set("${GENERATOR_GENERATED_HEADERS_VARIABLE}"
      ${LOCAL_GENERATED_HEADERS}
      PARENT_SCOPE)
  set("${GENERATOR_GENERATED_IMPLS_VARIABLE}"
      ${LOCAL_GENERATED_IMPLS}
      PARENT_SCOPE)
  set(EXTRA_TARGETS)

  #
  # Produce JSON schema, if requested
  #
  if(NOT "${GENERATOR_JSONSCHEMA_PATH}" STREQUAL "")
    tuple_tree_generator_generate_jsonschema(
      "${GENERATOR_SCHEMA_PATH}"
      "${GENERATOR_NAMESPACE}"
      "${GENERATOR_ROOT_TYPE}"
      "${GENERATOR_STRING_TYPES}"
      "${GENERATOR_SEPARATE_STRING_TYPES}"
      "${GENERATOR_SCALAR_TYPES}"
      "${GENERATOR_JSONSCHEMA_PATH}")
    list(APPEND EXTRA_TARGETS ${GENERATOR_JSONSCHEMA_PATH})
  endif()

  #
  # Produce Python code, if requested
  #
  if(NOT "${GENERATOR_PYTHON_PATH}" STREQUAL "")
    tuple_tree_generator_generate_python(
      "${GENERATOR_SCHEMA_PATH}"
      "${GENERATOR_NAMESPACE}"
      "${GENERATOR_ROOT_TYPE}"
      "${GENERATOR_STRING_TYPES}"
      "${GENERATOR_SEPARATE_STRING_TYPES}"
      "${GENERATOR_SCALAR_TYPES}"
      "${GENERATOR_PYTHON_PATH}")
    list(APPEND EXTRA_TARGETS ${GENERATOR_PYTHON_PATH})
  endif()

  #
  # Produce TypeScript code, if requested
  #
  if(NOT "${GENERATOR_TYPESCRIPT_PATH}" STREQUAL "")
    tuple_tree_generator_generate_typescript(
      "${GENERATOR_SCHEMA_PATH}"
      "${GENERATOR_NAMESPACE}"
      "${GENERATOR_ROOT_TYPE}"
      "${GENERATOR_GLOBAL_NAME}"
      "${GENERATOR_TYPESCRIPT_INCLUDE}"
      "${GENERATOR_STRING_TYPES}"
      "${GENERATOR_SEPARATE_STRING_TYPES}"
      "${GENERATOR_SCALAR_TYPES}"
      "${GENERATOR_TYPESCRIPT_PATH}")
    list(APPEND EXTRA_TARGETS ${GENERATOR_TYPESCRIPT_PATH})
  endif()

  add_custom_target(
    "${GENERATOR_TARGET_NAME}"
    DEPENDS "${GENERATOR_SCHEMA_PATH}" ${LOCAL_GENERATED_HEADERS}
            ${LOCAL_GENERATED_IMPLS} ${GENERATOR_JSONSCHEMA_PATH}
            ${EXTRA_TARGETS} generate-node_modules)
endfunction()

# Extracts tuple_tree_generator YAML definitions from the given header files
#
function(tuple_tree_generator_extract_definitions_from_headers HEADERS
         OUTPUT_FILE)
  add_custom_command(
    OUTPUT "${OUTPUT_FILE}"
    COMMAND "${CMAKE_SOURCE_DIR}/scripts/tuple_tree_generator/extract_yaml.py"
            --output "${OUTPUT_FILE}" TUPLE-TREE-YAML ${HEADERS}
    DEPENDS ${HEADERS})
endfunction()

# Computes the list of headers and C++ source files that will be generated by
# tuple_tree_generator. Note: the output variables will be overwritten
function(tuple_tree_generator_compute_generated_cpp_files SOURCE_HEADERS
         HEADERS_DIR GENERATED_HEADERS_VARIABLE GENERATED_IMPLS_VARIABLE)
  # Empty output variables
  set(LOCAL_GENERATED_HEADERS_VARIABLE "${HEADERS_DIR}/ForwardDecls.h")
  set(LOCAL_GENERATED_IMPLS_VARIABLE)

  foreach(HEADER ${SOURCE_HEADERS})
    # TODO: we should not be generating /Impl/*.cpp files in an include/*
    # directory
    #
    # TODO: this piece of code is tightly coupled with cppheaders.py
    get_filename_component(HEADER_FILENAME "${HEADER}" NAME)
    get_filename_component(HEADER_FILENAME_WE "${HEADER}" NAME_WE)
    set(EARLY_OUTPUT "${HEADERS_DIR}/Early/${HEADER_FILENAME}")
    set(LATE_OUTPUT "${HEADERS_DIR}/Late/${HEADER_FILENAME}")
    list(APPEND LOCAL_GENERATED_HEADERS_VARIABLE "${EARLY_OUTPUT}")
    list(APPEND LOCAL_GENERATED_HEADERS_VARIABLE "${LATE_OUTPUT}")
  endforeach()

  set(IMPL_OUTPUT "${HEADERS_DIR}/Impl.cpp")
  list(APPEND LOCAL_GENERATED_IMPLS_VARIABLE "${IMPL_OUTPUT}")

  set("${GENERATED_HEADERS_VARIABLE}"
      ${LOCAL_GENERATED_HEADERS_VARIABLE}
      PARENT_SCOPE)
  set("${GENERATED_IMPLS_VARIABLE}"
      ${LOCAL_GENERATED_IMPLS_VARIABLE}
      PARENT_SCOPE)
endfunction()

set(TEMPLATES_DIR
    "${CMAKE_SOURCE_DIR}/scripts/tuple_tree_generator/tuple_tree_generator/templates"
)

set(CPP_TEMPLATES
    "${TEMPLATES_DIR}/class_forward_decls.h.tpl"
    "${TEMPLATES_DIR}/enum.h.tpl"
    "${TEMPLATES_DIR}/struct.h.tpl"
    "${TEMPLATES_DIR}/struct_forward_decls.h.tpl"
    "${TEMPLATES_DIR}/struct_late.h.tpl"
    "${TEMPLATES_DIR}/struct_impl.cpp.tpl")

set(PYTHON_TEMPLATES "${TEMPLATES_DIR}/tuple_tree_gen.py.tpl")

set(TYPESCRIPT_TEMPLATES "${TEMPLATES_DIR}/tuple_tree_gen.ts.tpl")

set(SCRIPTS_ROOT_DIR "${CMAKE_SOURCE_DIR}/scripts/tuple_tree_generator")
# The list of Python scripts is build as follows:
#
# find scripts/tuple_tree_generator -name "*.py" | sort | sed
# 's|scripts/tuple_tree_generator|"\${SCRIPTS_ROOT_DIR}|; s/$/"/'
#
# TODO: detect and warn about extra files in those directories
set(TUPLE_TREE_GENERATOR_SOURCES
    "${SCRIPTS_ROOT_DIR}/extract_yaml.py"
    "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-cpp.py"
    "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-jsonschema.py"
    "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-python.py"
    "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-typescript.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/cppheaders.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/__init__.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/jinja_utils.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/jsonschema.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/python.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/generators/typescript.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/__init__.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/schema/definition.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/schema/enum.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/schema/__init__.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/schema/schema.py"
    "${SCRIPTS_ROOT_DIR}/tuple_tree_generator/schema/struct.py")

# Generates headers and implementation C++ files
function(
  tuple_tree_generator_generate_cpp
  # Path to the yaml definitions
  YAML_DEFINITIONS
  # Base namespace of the generated classes (e.g. model)
  NAMESPACE
  # Output directory
  OUTPUT_DIR
  # Include path prefix
  INCLUDE_PATH_PREFIX
  # List of headers that are expected to be generated
  EXPECTED_GENERATED_HEADERS
  # List of implementation files expected to be generated
  EXPECTED_GENERATED_IMPLS
  # Root type of the schema, if there is any
  ROOT_TYPE
  SCALAR_TYPES)

  set(SCALAR_TYPE_ARGS)
  foreach(ST ${SCALAR_TYPES})
    list(APPEND SCALAR_TYPE_ARGS --scalar-type "'${ST}'")
  endforeach()

  add_custom_command(
    COMMAND
      "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-cpp.py" --namespace
      "${NAMESPACE}" --include-path-prefix "${INCLUDE_PATH_PREFIX}" --root-type
      \""${ROOT_TYPE}"\" ${SCALAR_TYPE_ARGS} "${YAML_DEFINITIONS}"
      "${OUTPUT_DIR}"
    OUTPUT ${EXPECTED_GENERATED_HEADERS} ${EXPECTED_GENERATED_IMPLS}
    DEPENDS "${YAML_DEFINITIONS}" ${CPP_TEMPLATES}
            "${SCRIPTS_ROOT_DIR}/extract_yaml.py"
            ${TUPLE_TREE_GENERATOR_SOURCES})
endfunction()

# Generates JSON schema files
function(
  tuple_tree_generator_generate_jsonschema
  YAML_DEFINITIONS # Path to the yaml definitions
  NAMESPACE # Base namespace of the generated classes (e.g. model)
  ROOT_TYPE # Type to use as the root of the JSON schema
  STRING_TYPES # Types equivalent to plain strings
  SEPARATE_STRING_TYPES # Types equivalent to plain strings that get a separate
                        # type definition
  SCALAR_TYPES
  OUTPUT_PATH # Output path
)
  set(STRING_TYPE_ARGS)
  foreach(ST ${STRING_TYPES})
    list(APPEND STRING_TYPE_ARGS --string-type "${ST}")
  endforeach()

  set(SEPARATE_STRING_TYPE_ARGS)
  foreach(ST ${SEPARATE_STRING_TYPES})
    list(APPEND SEPARATE_STRING_TYPE_ARGS --separate-string-type "${ST}")
  endforeach()

  set(SCALAR_TYPE_ARGS)
  foreach(ST ${SCALAR_TYPES})
    list(APPEND SCALAR_TYPE_ARGS --scalar-type "'${ST}'")
  endforeach()

  add_custom_command(
    COMMAND
      "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-jsonschema.py" --namespace
      "${NAMESPACE}" --root-type "${ROOT_TYPE}" --output "${OUTPUT_PATH}"
      ${STRING_TYPE_ARGS} ${SEPARATE_STRING_TYPE_ARGS} ${SCALAR_TYPE_ARGS}
      "${YAML_DEFINITIONS}"
    OUTPUT "${OUTPUT_PATH}"
    DEPENDS "${YAML_DEFINITIONS}" ${TUPLE_TREE_GENERATOR_SOURCES})
endfunction()

# Generates typescript files
function(
  tuple_tree_generator_generate_typescript
  # Path to the yaml definitions
  YAML_DEFINITIONS
  # Base namespace of the generated classes (e.g. model)
  NAMESPACE
  # Type to use as the root of the schema
  ROOT_TYPE
  # Name of the global type, e.g. Model
  GLOBAL_NAME
  # Files to be included in the prouduced output
  INCLUDE_FILES
  # Types equivalent to plain strings
  STRING_TYPES
  # Types equivalent to plain strings that get a separate type definition
  EXTERNAL_TYPES
  SCALAR_TYPES
  # Output path
  OUTPUT_PATH)

  set(INCLUDE_FILES_ARGS)
  foreach(IF ${INCLUDE_FILES})
    list(APPEND INCLUDE_FILE_ARGS --external-file "${IF}")
  endforeach()

  set(STRING_TYPE_ARGS)
  foreach(ST ${STRING_TYPES})
    list(APPEND STRING_TYPE_ARGS --string-type "${ST}")
  endforeach()

  set(EXTERNAL_TYPE_ARGS)
  foreach(ET ${EXTERNAL_TYPES})
    list(APPEND EXTERNAL_TYPE_ARGS --external-type "${ET}")
  endforeach()

  set(SCALAR_TYPE_ARGS)
  foreach(ET ${SCALAR_TYPES})
    list(APPEND SCALAR_TYPE_ARGS --scalar-type "'${ET}'")
  endforeach()

  add_custom_command(
    COMMAND
      "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-typescript.py" --namespace
      "${NAMESPACE}" --root-type "${ROOT_TYPE}" --output "${OUTPUT_PATH}"
      --global-name "${GLOBAL_NAME}" --prettier
      "${CMAKE_BINARY_DIR}/node_build/node_modules/.bin/prettier"
      ${INCLUDE_FILE_ARGS} ${STRING_TYPE_ARGS} ${EXTERNAL_TYPE_ARGS}
      ${SCALAR_TYPE_ARGS} "${YAML_DEFINITIONS}"
    OUTPUT "${OUTPUT_PATH}"
    DEPENDS "${YAML_DEFINITIONS}" ${TYPESCRIPT_TEMPLATES}
            "${CMAKE_SOURCE_DIR}/typescript/model.ts"
            ${TUPLE_TREE_GENERATOR_SOURCES})
endfunction()

# Generates python files
function(
  tuple_tree_generator_generate_python
  # Path to the yaml definitions
  YAML_DEFINITIONS
  # Base namespace of the generated classes (e.g. model)
  NAMESPACE
  # Type to use as the root of the schema
  ROOT_TYPE
  # Types equivalent to plain strings
  STRING_TYPES
  # Types equivalent to plain strings that get a separate type definition
  EXTERNAL_TYPES
  SCALAR_TYPES
  # Output path
  OUTPUT_PATH)
  set(STRING_TYPE_ARGS)
  foreach(ST ${STRING_TYPES})
    list(APPEND STRING_TYPE_ARGS --string-type "${ST}")
  endforeach()

  set(EXTERNAL_TYPE_ARGS)
  foreach(ET ${EXTERNAL_TYPES})
    list(APPEND EXTERNAL_TYPE_ARGS --external-type "${ET}")
  endforeach()

  set(SCALAR_TYPE_ARGS)
  foreach(ET ${SCALAR_TYPES})
    list(APPEND SCALAR_TYPE_ARGS --scalar-type "'${ET}'")
  endforeach()

  add_custom_command(
    COMMAND
      "${SCRIPTS_ROOT_DIR}/tuple-tree-generate-python.py" --namespace
      "${NAMESPACE}" --root-type "${ROOT_TYPE}" --output "${OUTPUT_PATH}"
      ${STRING_TYPE_ARGS} ${EXTERNAL_TYPE_ARGS} ${SCALAR_TYPE_ARGS}
      "${YAML_DEFINITIONS}"
    OUTPUT "${OUTPUT_PATH}"
    DEPENDS "${YAML_DEFINITIONS}" ${PYTHON_TEMPLATES}
            ${TUPLE_TREE_GENERATOR_SOURCES})
endfunction()

# Extracts definitions and generates C++ headers and implementations from the
# given header files. The definitions must be embedded as described in the docs
# for tuple_tree_generator_extract_definitions_from_headers. Name of the target
# on which generated code will be attached too

# TARGET_ID HEADERS List of C++ headers

# NAMESPACE Delimiter used to mark comments embedding type schemas

# SCHEMA_PATH Where the schema will be collected

# HEADER_DIRECTORY Directory where the headers will be generated

# HEADER_DIRECTORY Full path where the headers will be generated, incompatible
# with INCLUDE_PATH_PREFIX

# HEADERS_PATH Include path prefix

# JSONSCHEMA_PATH Where the JSON schema will be produced (empty for no schema)

# ROOT_TYPE Type to use as the root of the JSON schema

# PYTHON_PATH  Path where the python generated code will be produced (empty for
# skipping python code generation)

# STRING_TYPES Types equivalent to strings

# SEPARATE_STRING_TYPES Types equivalent to strings which get a separate type
# definition
function(target_tuple_tree_generator TARGET_ID)
  set(options INSTALL)
  set(oneValueArgs
      HEADER_DIRECTORY
      NAMESPACE
      SCHEMA_PATH
      JSONSCHEMA_PATH
      ROOT_TYPE
      GLOBAL_NAME
      INCLUDE_PATH_PREFIX
      PYTHON_PATH
      TYPESCRIPT_PATH
      HEADERS_PATH)
  set(multiValueArgs HEADERS TYPESCRIPT_INCLUDE STRING_TYPES
                     SEPARATE_STRING_TYPES SCALAR_TYPES)
  cmake_parse_arguments(GEN "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        "${ARGN}")

  if(NOT DEFINED GEN_INCLUDE_PATH_PREFIX)
    set(GEN_INCLUDE_PATH_PREFIX "revng/${GEN_HEADER_DIRECTORY}")
  endif()

  # Generate C++ headers from the collected YAML
  #
  # TODO: the generated folder path should be configurable
  if(NOT DEFINED GEN_HEADERS_PATH)
    set(GEN_HEADERS_PATH
        "${CMAKE_BINARY_DIR}/include/revng/${GEN_HEADER_DIRECTORY}/Generated")
  endif()

  # Choose a target name that's available
  set(INDEX 1)
  set(GENERATOR_TARGET_NAME generate-${TARGET_ID}-tuple-tree-code)
  if(TARGET "${GENERATOR_TARGET_NAME}")
    math(EXPR INDEX "${INDEX}+1")
    set(GENERATOR_TARGET_NAME generate-${TARGET_ID}-tuple-tree-code-${INDEX})
  endif()

  tuple_tree_generator_impl(
    TARGET_NAME
    "${GENERATOR_TARGET_NAME}"
    HEADERS
    "${GEN_HEADERS}"
    NAMESPACE
    ${GEN_NAMESPACE}
    SCHEMA_PATH
    "${GEN_SCHEMA_PATH}"
    HEADERS_DIR
    "${GEN_HEADERS_PATH}"
    INCLUDE_PATH_PREFIX
    "${GEN_INCLUDE_PATH_PREFIX}"
    GENERATED_HEADERS_VARIABLE
    GENERATED_HEADERS
    GENERATED_IMPLS_VARIABLE
    GENERATED_IMPLS
    JSONSCHEMA_PATH
    "${GEN_JSONSCHEMA_PATH}"
    ROOT_TYPE
    ${GEN_ROOT_TYPE}
    GLOBAL_NAME
    ${GEN_GLOBAL_NAME}
    STRING_TYPES
    "${GEN_STRING_TYPES}"
    SEPARATE_STRING_TYPES
    "${GEN_SEPARATE_STRING_TYPES}"
    PYTHON_PATH
    ${GEN_PYTHON_PATH}
    TYPESCRIPT_PATH
    ${GEN_TYPESCRIPT_PATH}
    TYPESCRIPT_INCLUDE
    ${GEN_TYPESCRIPT_INCLUDE}
    SCALAR_TYPES
    ${GEN_SCALAR_TYPES})
  if(GEN_INSTALL)
    install(DIRECTORY ${GEN_HEADERS_PATH}
            DESTINATION include/revng/${GEN_HEADER_DIRECTORY})
  endif()

  target_sources(${TARGET_ID} PRIVATE ${GENERATED_IMPLS})

  add_dependencies(${TARGET_ID} "${GENERATOR_TARGET_NAME}")
endfunction()

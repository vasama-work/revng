#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(my-pass MyPass.cpp)
target_link_libraries(my-pass ${LLVM_LIBRARIES})

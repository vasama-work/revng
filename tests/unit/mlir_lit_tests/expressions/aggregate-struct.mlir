//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<SignedKind 4>

!s = !clift.defined<#clift.struct<
  id = 1,
  name = "",
  size = 1,
  fields = [
    <
      offset = 0,
      name = "x",
      type = !int32_t
    >,
    <
      offset = 4,
      name = "y",
      type = !int32_t
    >
  ]
>>

%0 = clift.imm 0 : !int32_t
%1 = clift.aggregate %0, %0 : (!int32_t, !int32_t) -> !s

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>

%m = clift.imm 0 : !int32_t
%c = clift.undef : !int32_t$const

clift.intrinsic "f"() : !clift.primitive<void 0>
clift.intrinsic "g"(%m : !int32_t, %c : !int32_t$const) : !int32_t

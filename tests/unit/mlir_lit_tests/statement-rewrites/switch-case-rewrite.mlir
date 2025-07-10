//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-statements="enable-patterns=switch-case-rewrite" | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
  } {
    // CHECK: [[L:%[0-9]+]] = clift.local !int32_t
    %0 = clift.local !int32_t

    // CHECK: clift.if {
    clift.switch {
      // CHECK: [[R1:%[0-9]+]] = clift.imm 1 : !int32_t
      // CHECK: [[R2:%[0-9]+]] = clift.eq [[L]], [[R1]] : !int32_t -> !int8_t
      clift.yield %0: !int32_t
      // CHECK: clift.yield [[R2]] : !int8_t
    // CHECK: } {
    } case 1 {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: [[R3:%[0-9]+]] = clift.imm 1 : !int32_t
        %1 = clift.imm 1 : !int32_t
        // CHECK: clift.yield [[R3]] : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    // CHECK: } else {
    } default {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: [[R4:%[0-9]+]] = clift.imm 2 : !int32_t
        %1 = clift.imm 2 : !int32_t
        // CHECK: clift.yield [[R4]] : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    // CHECK: }
    }

    // COM: Different condition region
    // CHECK: clift.if {
    clift.switch {
      // CHECK: [[L2:%[0-9]+]] = clift.imm 1 : !int32_t
      %1 = clift.imm 1 : !int32_t
      // CHECK: [[R5:%[0-9]+]] = clift.imm 3 : !int32_t
      // CHECK: [[R6:%[0-9]+]] = clift.eq [[L2]], [[R5]] : !int32_t -> !int8_t
      clift.yield %1: !int32_t
      // CHECK: clift.yield [[R6]] : !int8_t
    // CHECK: } {
    } case 3 {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: [[R7:%[0-9]+]] = clift.imm 1 : !int32_t
        %1 = clift.imm 1 : !int32_t
        // CHECK: clift.yield [[R7]] : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    // CHECK: } else {
    } default {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: [[R8:%[0-9]+]] = clift.imm 2 : !int32_t
        %1 = clift.imm 2 : !int32_t
        // CHECK: clift.yield [[R8]] : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    // CHECK: }
    }

    // COM: This shouldn't get rewritten
    // CHECK: clift.switch {
    clift.switch {
      clift.yield %0: !int32_t
    // CHECK: } case 1 {
    } case 1 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    // CHECK: } case 2 {
    } case 2 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    // CHECK: } default {
    } default {
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    // CHECK: }
    }
  }
  // CHECK: }
}

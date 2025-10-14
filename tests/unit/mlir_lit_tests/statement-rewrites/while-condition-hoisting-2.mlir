//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-statements="enable-patterns=while-condition-hoisting" | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>
!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.while
    // CHECK-SAME: cond {
    clift.while cond {
      %1 = clift.imm 1 : !int32_t
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %1 : !int32_t
    // CHECK: } body {
    } body {
      // CHECK-NOT: clift.if
      clift.if {
        clift.yield %arg0 : !int32_t
      } then {
        // CHECK-NEXT: clift.expr {
        clift.expr {
          // CHECK: %1 = clift.imm 2 : !int32_t
          %1 = clift.imm 2 : !int32_t
          // CHECK: clift.yield %1 : !int32_t
          clift.yield %1 : !int32_t
        // CHECK: }
        }
      } else {
        clift.goto %0
      }
      // CHECK-NEXT: clift.expr {
      clift.expr {
        // CHECK: %1 = clift.imm 3 : !int32_t
        %1 = clift.imm 3 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
    }
    // CHECK: clift.assign_label %0
    clift.assign_label %0
  // CHECK: }
  }
// CHECK: }
}

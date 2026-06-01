//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: for (;;)
    clift.for body {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        %2 = clift.discard %1 : !int32_t
        clift.yield %2 : !void
      }
    }

    // CHECK: for (int32_t x = 2; x; x) {
    clift.for init : !int32_t {
      clift.local : !int32_t = {
        %2 = clift.imm 2 : !int32_t
        clift.yield %2 : !int32_t
      } attributes {
        handle = "/local-variable/0x40001001:Code_x86_64/0",
        name = "x"
      }
    } cond (%x) {
      clift.yield %x : !int32_t
    } next (%x) {
      clift.yield %x : !int32_t
    } body (%x) {
      // CHECK: x;
      clift.expr {
        %1 = clift.discard %x : !int32_t
        clift.yield %1 : !void
      }
      // CHECK: 3;
      clift.expr {
        %5 = clift.imm 3 : !int32_t
        %6 = clift.discard %5 : !int32_t
        clift.yield %6 : !void
      }
    }
    // CHECK: }
  }
  // CHECK: }
}

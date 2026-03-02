//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --eliminate-implicit-casts | FileCheck %s

!void = !clift.primitive<void 0>
!uint8_t = !clift.primitive<unsigned 1>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!uint8_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uint8_t) -> !void {
    // x = (uint8_t)(                                                          )
    //               (int32_t)x / (int32_t)(                                  )
    //                                      (uint8_t)(                       )
    //                                                (int32_t)x + (int32_t)x
    //
    // x = x / (uint8_t)(x + x);

    // CHECK: clift.expr {
    clift.expr {
      // CHECK-DAG: %[[INNER_LHS:[0-9]+]] = clift.cast<extend> %arg0 {clift.implicit} : !uint8_t -> !int32_t
      // CHECK-DAG: %[[INNER_RHS:[0-9]+]] = clift.cast<extend> %arg0 {clift.implicit} : !uint8_t -> !int32_t
      // CHECK-DAG: %[[INNER_ADD:[0-9]+]] = clift.add %[[INNER_LHS]], %[[INNER_RHS]] : !int32_t

      // CHECK-DAG: %[[OUTER_LHS:[0-9]+]] = clift.cast<extend> %arg0 {clift.implicit} : !uint8_t -> !int32_t
      // CHECK-DAG: %[[INNER_VAL:[0-9]+]] = clift.cast<truncate> %[[INNER_ADD]] : !int32_t -> !uint8_t
      // CHECK-DAG: %[[OUTER_RHS:[0-9]+]] = clift.cast<extend> %[[INNER_VAL]] {clift.implicit} : !uint8_t -> !int32_t
      // CHECK-DAG: %[[OUTER_DIV:[0-9]+]] = clift.div %[[OUTER_LHS]], %[[OUTER_RHS]] : !int32_t

      // CHECK-DAG: %[[ASSIGN_RHS:[0-9]+]] = clift.cast<truncate> %[[OUTER_DIV]] {clift.implicit} : !int32_t -> !uint8_t
      // CHECK: %[[ASSIGN:[0-9]+]] = clift.assign %arg0, %[[ASSIGN_RHS]] : !uint8_t
      // CHECK: clift.yield %[[ASSIGN]] : !uint8_t

      %0 = clift.add %arg0, %arg0 : !uint8_t
      %1 = clift.div %arg0, %0 : !uint8_t
      %2 = clift.assign %arg0, %1 : !uint8_t
      clift.yield %2 : !uint8_t
    }
    // CHECK: }
  }
}

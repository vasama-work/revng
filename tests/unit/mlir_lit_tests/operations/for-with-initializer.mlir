//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

clift.for init : !int32_t {
  clift.local : !int32_t
} body (%i) {
  clift.expr {
    clift.yield %i : !int32_t
  }
}

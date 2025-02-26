//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

// CHECK: unique handle must be non-empty
clift.intrinsic ""() : !clift.primitive<VoidKind 0>

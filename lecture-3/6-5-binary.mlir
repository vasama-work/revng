// 1 + 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.add %0, %1 : !int32_t
  clift.yield %2 : !int32_t
}

// add: x + y
// sub: x - y
// mul: x * y
// div: x / y
// rem: x % y

// 1 & 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.bitand %0, %1 : !int32_t
  clift.yield %2 : !int32_t
}

// bitand: x & y
// bitor:  x | y
// bitxor: x ^ y

// 1 << 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.shl %0, %1 : !int32_t
  clift.yield %2 : !int32_t
}

// shl: x << y
// shr: x >> y

// Unlike other similar operations, shifts accept different operand types.
// The result type matches the left operand type.

// 1 == 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.eq %0, %1 : !int32_t -> !int8_t
  clift.yield %2 : !int8_t
}

// eq: x == y
// ne: x != y
// lt: x < y
// gt: x > y
// le: x <= y
// ge: x >= y

%p = clift.local : !clift.ptr<8 to !int32_t>

// p + 1LL;
clift.expr {
  %0 = clift.immediate 1 : !uint64_t
  %1 = clift.ptr_add %p, %0 : (!clift.ptr<8 to !int32_t>, !uint64_t)
  clift.yield %1 : !clift.ptr<8 to !int32_t>
}

// ptr_add:  p + 1
//           1 + p
// ptr_sub:  p - 1

// p - p
clift.expr {
  %0 = clift.ptr_diff %p, %p : !clift.ptr<8 to !int32_t> -> !int64_t
  clift.yield %0 : !int64_t
}

// 1 && 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.and %0, %1 : !int32_t -> !int8_t
  clift.yield %2 : !int8_t
}

// and: x && y
// or:  x || y

// 1, 2;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.immediate 2 : !int32_t
  %2 = clift.comma %0, %1 : !int32_t, !int32_t
  clift.yield %2 : !int32_t
}

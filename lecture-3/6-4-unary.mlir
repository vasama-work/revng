// -42;
clift.expr {
  %0 = clift.immediate 42 : !int32_t
  %1 = clift.neg %0 : !int32_t
  clift.yield %1 : !int32_t
}

// !42;
clift.expr {
  %0 = clift.immediate 42 : !int32_t
  %1 = clift.not %0 : !int32_t -> !int8_t
  clift.yield %1 : !int8_t
}

// ~42;
clift.expr {
  %0 = clift.immediate 42 : !int32_t
  %1 = clift.bitnot %0 : !int32_t
  clift.yield %1 : !int32_t
}

%x = clift.local : !int32_t
%p = clift.local : !clift.ptr<8 to !int32_t>

// &x;
clift.expr {
  %0 = clift.addressof %x : !clift.ptr<8 to !int32_t>
  clift.yield %0 : !clift.ptr<8 to !int32_t>
}

// *p;
clift.expr {
  %0 = clift.indirection %p : !clift.ptr<8 to !int32_t>
  clift.yield %0 : !int32_t
}

// (uint32_t)x;
clift.expr {
  %0 = clift.cast<bitcast> %x : !int32_t -> !uint32_t
  clift.yield %0 : !uint32_t
}

// * clift.cast<extend>
//   Converts a narrower integer to a wider integer.
//
// * clift.cast<truncate>
//   Converts a wider integer to a narrower integer.
//
// * clift.cast<bitcast>
//   Converts between any two types of matching size.
//
// * clift.cast<decay>
//   Converts array and function values to pointers.
//
// * clift.cast<convert>
//   Converts between integer and floating point types.
//

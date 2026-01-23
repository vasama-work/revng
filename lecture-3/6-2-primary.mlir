// undef(int32_t)
clift.expr {
  %0 = clift.undef : !int32_t
  clift.yield %0 : !int32_t
}

// 42;
clift.expr {
  %0 = clift.immediate 42 : !int32_t
  clift.yield %0 : !int32_t
}

// X;
clift.expr {
  // Assume !enum has enumerator `X = 42`.

  %0 = clift.immediate 42 : !enum
  clift.yield %0 : !enum
}

// "hello";
clift.expr {
  %0 = clift.string "hello" : !clift.array<!int8_t x 6>
}

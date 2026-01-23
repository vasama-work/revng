%x = clift.local : !int32_t

// x = 1;
clift.expr {
  %0 = clift.immediate 1 : !int32_t
  %1 = clift.assign %x, %0 : !int32_t
  clift.yield %1 : !int32_t
}

// ++x;
clift.expr {
  %0 = clift.inc %x : !int32_t
  clift.yield %0 : !int32_t
}

//      inc: ++x
// post_inc: x++
//      dec: --x
// post_dec: x--

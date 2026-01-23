%array = clift.local : !clift.array<!int32_t x 10>

// array[0];
clift.expr {
  %0 = clift.cast<decay> %array : !clift.array<!int32_t x 10> -> !clift.ptr<8 to !int32_t>
  %1 = clift.immediate 0 : !int64_t
  %2 = clift.subscript %0, %1 : (!clift.ptr<8 to !int32_t>, !int64_t)
  clift.yield %2 : !int32_t
}

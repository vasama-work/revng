!my_struct = !clift.struct<
  "/type-definition/1001-StructDefinition" as "my_struct" : size(8) {
    "/struct-field/1001-StructDefinition/0" as "x" : offset(0) !int32_t,
    "/struct-field/1001-StructDefinition/1" as "y" : offset(4) !int32_t
  }
>

// (my_struct){ 1, 2 };
clift.expr {
  %0 = clift.imm 1 : !int32_t
  %1 = clift.imm 2 : !int32_t
  %2 = clift.aggregate (%0, %1) : !my_struct
  clift.yield %2 : !my_struct
}

%s = clift.local : !my_struct
%p = clift.local : !clift.ptr<8 to !my_struct>

// s.x;
clift.expr {
  %0 = clift.access<0> %s : !my_struct -> !int32_t
  clift.yield %0 : !int32_t
}

// p->x;
clift.expr {
  %0 = clift.access<indirect 0> %p : !clift.ptr<8 to !my_struct> -> !int32_t
  clift.yield %0 : !int32_t
}

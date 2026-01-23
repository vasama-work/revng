!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // int32_t local_0;
    %0 = clift.local : !int32_t


    // int32_t local_1 = arg0;
    %1 = clift.local : !int32_t = {
      clift.yield %arg0 : !int32_t
    }


  }
}

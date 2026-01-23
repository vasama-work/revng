!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // return;
    clift.return {}


    // return arg0;
    clift.return {
      clift.yield %arg0 : !int32_t
    }


  }
}

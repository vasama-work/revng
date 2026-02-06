!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // my_global;
    clift.expr {
      %my_global = clift.use @my_global : !int32_t
      clift.yield %my_global : !int32_t
    }


    // my_function(arg0);
    clift.expr {
      %my_function = clift.use @my_function : !my_function
      %0 = clift.call %my_function(%arg0) : !my_function
      clift.yield %0 : !void
    }


  }
}

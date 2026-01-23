!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // while (arg0) {}
    clift.while cond {
      clift.yield %arg0 : !int32_t
    } body {
      // Statement region:
    }


    // do {} while (arg0);
    clift.do_while body {
      // Statement region:
    } cond {
      clift.yield %arg0 : !int32_t
    }


    // for (;;) {}
    clift.for body {
      // Statement region:
    }


  }
}

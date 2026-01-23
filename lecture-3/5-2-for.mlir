!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // for (;;)
    clift.for body {
      // Statement region:
    }

    "clift.for"() ({
      // init
    }, {
      // cond
    }, {
      // next
    }, {
      // body
    }) : () -> ()


    // for (arg0; arg0; arg0) {}
    clift.for init {
      clift.expr {
        clift.yield %arg0 : !int32_t
      }
    } cond {
      clift.yield %arg0 : !int32_t
    } next {
      clift.yield %arg0 : !int32_t
    } body {
      // Statement region:
    }


    // for (int32_t i = arg0; i; i) { i; }
    clift.for init : !int32_t {
      clift.local : !int32_t = {
        clift.yield %arg0 : !int32_t
      }
    } cond (%i) {
      clift.yield %i : !int32_t
    } next (%i) {
      clift.yield %i : !int32_t
    } body (%i) {
      clift.expr {
        clift.yield %i : !int32_t
      }
    }

    "clift.for"() ({
      // init

      clift.local : !int32_t = {
        clift.yield %arg0 : !int32_t
      }

    }, {
      // cond

    ^0(%i : !int32_t):
      clift.yield %arg0 : !int32_t

    }, {
      // next

    ^0(%i : !int32_t):
      clift.yield %arg0 : !int32_t

    }, {
      // body

    ^0(%i : !int32_t):
      clift.expr {
        clift.yield %i : !int32_t
      }

    }) : () -> ()


  }
}

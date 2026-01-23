!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    // if (arg0) {}
    clift.if {
      // Expression region:
      clift.yield %arg0 : !int32_t
    } then {
      // Statement region:
    } else {
      // Statement region:
    }

    "clift.if"() ({
      clift.yield %arg0 : !int32_t
    }, {
      // then
    }, {
      // else
    }) : () -> ()


    // switch (arg0) {}
    clift.switch {
      // Expression region:
      clift.yield %arg0 : !int32_t
    } case 0 {
      // Statement region:
    } case 42 {
      // Statement region:
    } default {
      // Statement region:
    }

    "clift.switch"() ({
      clift.yield %arg0 : !int32_t
    }, {
      // default
    }, {
      // case 0
    }, {
      // case 42
    }) {
      case_values = [0, 42]
    } : () -> ()


  }
}

!int32_t = !clift.primitive<signed 4>

!my_function = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "my_function" : !int32_t(!int32_t)
>

module attributes { clift.module } {
  clift.global @my_global : !int32_t

  // void my_function(int32_t arg0) {
  clift.func @my_function<!my_function>(%arg0 : !int32_t) {


    %break = clift.make_label
    %continue = clift.make_label

    // for (;;) {
    //  // Loop content:
    //  // ...
    // loop_continue:
    // }
    // loop_break:
    clift.for break %break continue %continue body {

      // break_to loop_break;
      clift.break_to %break

      // continue_to loop_continue;
      clift.continue_to %continue
    }

    // goto loop_break;
    clift.goto %break

    // goto loop_continue;
    clift.goto %continue


    clift.for body {
      // Loop content:
      // ...

      clift.assign_label %continue
    }
    clift.assign_label %break


  }
}

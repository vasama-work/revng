void doSomething(int x) {

  x; // int; lvalue (may appear on the *left* side of an assignment).

  0; // int; rvalue (may only appear on the *right* side of an assignment).

  //   valid: ++x
  // invalid: ++0

  //   valid: x = 0
  // invalid: 0 = 0

  //   valid: &x
  // invalid: &0

}

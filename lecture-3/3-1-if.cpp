void doSomething(clift::IfOp If) {

  mlir::Region &Cond = If.getCondition();
  mlir::Region &Then = If.getThen();
  mlir::Region &Else = If.getElse();

}

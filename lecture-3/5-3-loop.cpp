void doSomething(clift::LoopOpInterface Loop) {

  mlir::Value BreakLabel = Loop.getBreakLabel();
  Loop.setBreakLabel(BreakLabel);

  mlir::Value ContinueLabel = Loop.getContinueLabel();
  Loop.setContinueLabel(ContinueLabel);

  mlir::Region &Body = Loop.getBody();

}

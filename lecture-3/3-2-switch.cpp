void doSomething(clift::SwitchOp Switch) {

  mlir::Region &Cond = Switch.getCondition();

  bool HasDefaultCase = Switch.hasDefaultCase();
  /* Equivalent to: */  Switch.getDefault().empty();

  unsigned NumCases = Switch.getNumCases();

  for (unsigned I = 0; I < NumCases; ++I) {
    uint64_t CaseValue = Switch.getCaseValue(I);
    mlir::Region &CaseRegion = Switch.getCaseRegion(I);
  }

  for (mlir::Region &CaseRegion : Switch.getCaseRegions()) {
  }

  mlir::Region *R = Switch.findCaseRegion(42);

}

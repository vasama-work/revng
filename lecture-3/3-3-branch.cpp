void doSomething(clift::BranchOpInterface Branch) {

  // Condition is an expression region.
  // Not necessarily boolean-tested.
  mlir::Region &Condition = Branch.getConditionRegion();

  // Exposes the full set of potential branches, including else and default.
  for (mlir::Region &Region : Branch.getBranchRegions()) {
    // Region is a statement region.
  }

}

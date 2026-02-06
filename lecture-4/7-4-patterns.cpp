struct PatternRewritingPass : impl::MyPatternRewritingBase<PatternRewritingPass> {

  mlir::FrozenRewritePatternSet Patterns;

  // Optional pass initialization step:
  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {

    mlir::RewritePatternSet Set(Context);

    // Add available rewrite patterns to the set ...

    // Create an immutable set and apply filtering based on pass options:
    //
    // disabledPatterns and enabledPatterns are memebrs of the CRTP base class,
    // and are populated from pass options (possibly from the command line).
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {

    FunctionOp Function = getOperation();

    // Iteratively apply all rewrites in the set:
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns).failed())
      signalPassFailure();

  }

};

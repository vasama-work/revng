void doSomething(mlir::Operation *Op) {

  for (mlir::Region &Region : Op->getRegions()) {
    // Iterates over each region in the operation.
    // Usually regions of a specific type of operation are named.

    for (mlir::Block &Block : Region.getBlocks()) {
      // Iterates over each block in the region.

      for (mlir::Operation &Op : Block.getOperations()) {
        // Iterates over each operation in the block.

        doSomething(&Op);
      }
    }
  }
}

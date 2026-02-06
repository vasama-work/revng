#include "revng/CliftTransforms/Passes.h.inc"

// Pass implementation, derived from CRTP base class:
struct LoopDetectionPass : impl::CliftLoopDetectionBase<LoopDetectionPass> {

  void runOnOperation() override {

    // clift::FunctionOp
    auto Op = getOperation();

    // ...


  }

};

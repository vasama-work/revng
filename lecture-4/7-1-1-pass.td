include "mlir/Pass/PassBase.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftLoopDetection : Clift_Pass<"loop-detection", "clift::FunctionOp"> {
  let summary = "Detect Clift loops";
  let constructor = "mlir::clift::createLoopDetectionPass()";
}

def CliftEmitC : Clift_Pass<"emit-c", "mlir::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";

  let options = [
    Option<"EmitTags", "ptml", "bool", /*default=*/"false", "Emit PTML tags">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

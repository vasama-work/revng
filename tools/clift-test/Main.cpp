//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/Transforms/ModelOption.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"

namespace clift = mlir::clift;
namespace cl = llvm::cl;

using namespace clift;

int main(int Argc, char **Argv) {
  static cl::opt<std::string> InputFilename(cl::Positional,
                                            cl::desc("<input file>"),
                                            cl::init("-"));

  static cl::opt<clift::ModelOptionType> Model("model",
                                               cl::desc("Model file path"));

  static cl::opt<bool> Tagless("tagless", cl::init(false));

  llvm::InitLLVM Init(Argc, Argv);

  // NOLINTNEXTLINE
  cl::ParseCommandLineOptions(Argc, Argv, "clift-emit");

  std::string ErrorMessage;
  auto InputFile = mlir::openInputFile(InputFilename, &ErrorMessage);
  if (!InputFile) {
    llvm::errs() << ErrorMessage << "\n";
    return EXIT_FAILURE;
  }

  mlir::DialectRegistry Registry;
  Registry.insert<CliftDialect>();

  mlir::MLIRContext Context(mlir::MLIRContext::Threading::DISABLED);
  Context.appendDialectRegistry(Registry);
  Context.loadAllAvailableDialects();

  auto ParserConfig = mlir::ParserConfig(&Context);
  auto Module = mlir::parseSourceString<clift::ModuleOp>(InputFile->getBuffer(),
                                                         ParserConfig);

  if (not Module)
    return EXIT_FAILURE;

  clift::PlatformInfo Platform = {
    .sizeof_char = 1,
    .sizeof_short = 2,
    .sizeof_int = 4,
    .sizeof_long = 8,
    .sizeof_longlong = 8,
    .sizeof_float = 4,
    .sizeof_double = 8,
    .sizeof_pointer = 8,
  };

  std::string Result;
  llvm::raw_string_ostream Out(Result);

  ptml::CTypeBuilder B(Out, *Model, ptml::CBuilder(Tagless));

  Module->walk([&](FunctionOp F) {
    if (not F.isExternal())
      printf("%s\n", clift::decompile(F, Platform, B).c_str());
  });

  return EXIT_SUCCESS;
}
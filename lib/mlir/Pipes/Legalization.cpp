//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/Legalization.h"
#include "revng/mlir/Pipes/CliftContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class CBackendPipe {
public:
  static constexpr auto Name = "clift-legalization";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace kinds;

    return { ContractGroup({ Contract(MLIRFunctionKind,
                                      0,
                                      MLIRFunctionKind,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           pipes::CliftContainer &CliftContainer) {

    // TODO: Store this information in the model or another configuration.
    clift::TargetCImplementation Target = {
      .PointerSize = 8,
      .IntegerTypes = {
        { 1, clift::CIntegerKind::Char },
        { 2, clift::CIntegerKind::Short },
        { 4, clift::CIntegerKind::Int },
        { 8, clift::CIntegerKind::Long },
      },
    };

    mlir::ModuleOp Module = CliftContainer.getModule();

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    for (const model::Function &Function :
         getFunctionsAndCommit(EC, CliftContainer.name())) {
      auto It = Functions.find(Function.Entry());
      revng_check(It != Functions.end()
                  and "Requested Clift function not found");

      revng_check(legalizeForC(It->second, Target).succeeded());
    }
  }
};

static pipeline::RegisterPipe<CBackendPipe> X;

} // namespace

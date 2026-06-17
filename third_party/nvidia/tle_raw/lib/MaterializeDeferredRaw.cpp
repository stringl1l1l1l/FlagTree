#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "nvidia/tle_raw/include/DeferredRawSourceRegistry.h"
#include "nvidia/tle_raw/include/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace tle = mlir::triton::tle;
namespace nv_tle_raw = mlir::triton::nvidia::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_NVIDIAMATERIALIZEDEFERREDRAW
#include "nvidia/tle_raw/include/Passes.h.inc"

class NvidiaMaterializeDeferredRawPass
    : public impl::NvidiaMaterializeDeferredRawBase<
          NvidiaMaterializeDeferredRawPass> {
public:
  using impl::NvidiaMaterializeDeferredRawBase<
      NvidiaMaterializeDeferredRawPass>::NvidiaMaterializeDeferredRawBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &registry = nv_tle_raw::getDeferredRawSourceRegistry();

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result = module.walk([&](tle::DSLRegionOp op) -> WalkResult {
      auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
      if (!sourceIdAttr)
        return WalkResult::advance();

      auto it = registry.find(sourceIdAttr.getValue());
      if (it == registry.end()) {
        op.emitError("missing pending raw source for id ")
            << sourceIdAttr.getValue();
        return WalkResult::interrupt();
      }

      const nv_tle_raw::DeferredRawSourceEntry &entry = it->second;
      (void)entry;
      // TODO: clang CUDA -> MLIR LLVM dialect body, then splice into dsl_region.
      op.emitError("deferred raw materialization is not implemented yet");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
    nv_tle_raw::clearDeferredRawSourceRegistry();
  }
};

} // namespace mlir

#include "tle/dialect/include/Conversion/TleToLLVM/GetLocalPeOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

static LLVM::LLVMFuncOp getLocalPeFuction(ModuleOp module, MLIRContext *ctx) {

  const char *funcName = "flagcxDevCommGetIntraRank";
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  auto funcType = LLVM::LLVMFunctionType::get(PtrTy, {PtrTy}, false);

  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);

  func.setLinkage(LLVM::Linkage::External);
  return func;
}

struct GetLocalPeOpConversion
    : public ConvertOpToLLVMPattern<tle::GetLocalPeOp> {
  GetLocalPeOpConversion(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetLocalPeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value memPtr = srcElems[0];
    ModuleOp module = rewriter.getInsertionPoint()
                          ->getParentOp()
                          ->getParentOfType<ModuleOp>();

    if (!module) {
      return rewriter.notifyMatchFailure(loc, "expected module context");
    }
    auto func = getLocalPeFuction(module, rewriter.getContext());
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);

    Value comm_dev_ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, memPtr);
    auto getLocalPeCall = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{func.getFunctionType().getReturnType()},
        FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});

    Value my_pe = getLocalPeCall.getResult();
    rewriter.replaceOp(op, my_pe);
    return success();

    op.dump();

    llvm::errs() << "[GetLocalPeOpConversion]";
    return success();
  }
};

} // namespace
void tle::populateGetLocalPeOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             PatternBenefit benefit) {
  patterns.add<GetLocalPeOpConversion>(typeConverter, benefit);
}

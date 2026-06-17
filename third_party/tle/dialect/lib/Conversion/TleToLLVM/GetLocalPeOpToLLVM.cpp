#include "tle/dialect/include/Conversion/TleToLLVM/GetLocalPeOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

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

struct GetNumPesOpConversion : public ConvertOpToLLVMPattern<tle::GetNumPesOp> {
  GetNumPesOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetNumPesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto getLocalPeCall = tle::getNumPesFunCall(loc, rewriter, srcElems[0]);

    Value n_pes = getLocalPeCall.getResult();
    rewriter.replaceOp(op, n_pes);
    return success();

    op.dump();

    llvm::errs() << "[GetLocalPeOpConversion]";
    return success();
  }
};

struct GetLocalPeOpConversion
    : public ConvertOpToLLVMPattern<tle::GetLocalPeOp> {
  GetLocalPeOpConversion(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetLocalPeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto getLocalPeCall = tle::getLocalPeFuncCall(loc, rewriter, srcElems[0]);

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

void tle::populateGetNumPesOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<GetNumPesOpConversion>(typeConverter, benefit);
}

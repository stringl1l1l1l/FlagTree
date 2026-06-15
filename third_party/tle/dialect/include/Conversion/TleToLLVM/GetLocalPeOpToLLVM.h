#ifndef TLE_CONVERSION_TLETOLLVM_GETLOCALPEOPTOLLVM_H
#define TLE_CONVERSION_TLETOLLVM_GETLOCALPEOPTOLLVM_H

#include "mlir/IR/Value.h"

namespace mlir::triton::tle {

unsigned inferTlePointerLayoutVectorHint(Value ptr);

} // namespace mlir::triton::tle

#endif

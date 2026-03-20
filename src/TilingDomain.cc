#include "src/TilingDomain.h"

#include "src/TilingDomain.cpp.inc"

#define GET_OP_CLASSES
#include "src/TilingDomainOps.cpp.inc"

namespace mlir {
namespace tiling_domain {

void TilingDomainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/TilingDomainOps.cpp.inc"
      >();
}

} // namespace tiling_domain
} // namespace mlir
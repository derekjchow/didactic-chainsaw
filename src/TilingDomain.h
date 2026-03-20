#ifndef SRC_TILING_DOMAIN_H_
#define SRC_TILING_DOMAIN_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/TilingDomain.h.inc"

#define GET_OP_CLASSES
#include "src/TilingDomainOps.h.inc"

#endif  // SRC_TILING_DOMAIN_H_
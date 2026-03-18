#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tfl_stablehlo_pass.h"
#include "llvm/Support/MemoryBuffer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_file>" << std::endl;
    return 1;
  }

  std::string input_file = argv[1];

  mlir::MLIRContext context;
  context.loadDialect<mlir::TFL::TFLDialect, mlir::linalg::LinalgDialect,
                      mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  std::string error_message;
  auto buffer = mlir::openInputFile(input_file, &error_message);
  if (!buffer) {
    std::cerr << "Failed to open input file: " << error_message << std::endl;
    return 1;
  }

  // Import TFLite FlatBuffer to TFL Dialect MLIR.
  // We use absl::string_view for the buffer.
  mlir::OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      &context, mlir::UnknownLoc::get(&context));

  if (!module) {
    std::cerr << "Failed to import FlatBuffer to MLIR" << std::endl;
    return 1;
  }

  // Lower from TFL Dialect to TOSA and then to Linalg.
  mlir::PassManager pm(&context);
  
  // TFLite -> StableHLO
  pm.addPass(mlir::odml::CreateTflToStablehloPass());
  // StableHLO -> Linalg
  pm.addNestedPass<mlir::func::FuncOp>(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to lower to Linalg" << std::endl;
    return 1;
  }

  module->print(llvm::outs());

  return 0;
}

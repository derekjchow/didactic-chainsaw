#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/MemoryBuffer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_file>" << std::endl;
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TFLDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string error_message;
  auto buffer = mlir::openInputFile(argv[1], &error_message);
  if (!buffer) {
    std::cerr << "Failed to open input file: " << error_message << std::endl;
    return 1;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      &context, mlir::UnknownLoc::get(&context));

  if (!module) {
    std::cerr << "Failed to import FlatBuffer to MLIR" << std::endl;
    return 1;
  }

  module->print(llvm::outs());

  return 0;
}

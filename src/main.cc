#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/MemoryBuffer.h"

namespace {

// Rewriter to convert tfl.conv_2d to linalg.generic.
struct Conv2DRewriter : public mlir::OpConversionPattern<mlir::TFL::Conv2DOp> {
  using OpConversionPattern<mlir::TFL::Conv2DOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::Conv2DOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!outputType) return mlir::failure();

    // Handle dynamic dimensions for the output tensor.
    llvm::SmallVector<mlir::Value, 4> dynamicSizes;
    for (int i = 0; i < outputType.getRank(); ++i) {
      if (outputType.isDynamicDim(i)) {
        dynamicSizes.push_back(rewriter.create<mlir::tensor::DimOp>(op.getLoc(), adaptor.getInput(), i));
      }
    }
    
    auto emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType(), dynamicSizes);

    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto filter = adaptor.getFilter();
    
    // Indexing maps for NHWC convolution: [n, h, w, f, kh, kw, c]
    auto n = rewriter.getAffineDimExpr(0);
    auto h = rewriter.getAffineDimExpr(1);
    auto w = rewriter.getAffineDimExpr(2);
    auto f = rewriter.getAffineDimExpr(3);
    auto kh = rewriter.getAffineDimExpr(4);
    auto kw = rewriter.getAffineDimExpr(5);
    auto c = rewriter.getAffineDimExpr(6);

    int64_t strideH = op.getStrideH();
    int64_t strideW = op.getStrideW();
    int64_t dilationH = op.getDilationHFactor();
    int64_t dilationW = op.getDilationWFactor();

    auto inputMap = mlir::AffineMap::get(7, 0, {n, h * strideH + kh * dilationH, w * strideW + kw * dilationW, c}, rewriter.getContext());
    auto filterMap = mlir::AffineMap::get(7, 0, {f, kh, kw, c}, rewriter.getContext());
    auto outputMap = mlir::AffineMap::get(7, 0, {n, h, w, f}, rewriter.getContext());

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(7, mlir::utils::IteratorType::parallel);
    iteratorTypes[4] = mlir::utils::IteratorType::reduction; // kh
    iteratorTypes[5] = mlir::utils::IteratorType::reduction; // kw
    iteratorTypes[6] = mlir::utils::IteratorType::reduction; // c

    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, outputType, 
        mlir::ValueRange{input, filter}, 
        mlir::ValueRange{emptyTensor.getResult()},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, filterMap, outputMap},
        iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            // Placeholder: yield output to preserve structural conversion.
            // Full quantization support would involve dequantize -> compute -> quantize logic.
            b.create<mlir::linalg::YieldOp>(loc, args[2]);
        });

    rewriter.replaceOp(op, genericOp->getResults());
    return mlir::success();
  }
};

// Rewriter to convert tfl.depthwise_conv_2d to linalg.generic.
struct DepthwiseConv2DRewriter : public mlir::OpConversionPattern<mlir::TFL::DepthwiseConv2DOp> {
  using OpConversionPattern<mlir::TFL::DepthwiseConv2DOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::DepthwiseConv2DOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!outputType) return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> dynamicSizes;
    for (int i = 0; i < outputType.getRank(); ++i) {
      if (outputType.isDynamicDim(i)) {
        dynamicSizes.push_back(rewriter.create<mlir::tensor::DimOp>(op.getLoc(), adaptor.getInput(), i));
      }
    }
    
    auto emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType(), dynamicSizes);

    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto filter = adaptor.getFilter();
    
    // Indexing maps for NHWC Depthwise: [n, h, w, c, kh, kw]
    auto n = rewriter.getAffineDimExpr(0);
    auto h = rewriter.getAffineDimExpr(1);
    auto w = rewriter.getAffineDimExpr(2);
    auto c = rewriter.getAffineDimExpr(3);
    auto kh = rewriter.getAffineDimExpr(4);
    auto kw = rewriter.getAffineDimExpr(5);

    int64_t strideH = op.getStrideH();
    int64_t strideW = op.getStrideW();
    int64_t dilationH = op.getDilationHFactor();
    int64_t dilationW = op.getDilationWFactor();

    auto inputMap = mlir::AffineMap::get(6, 0, {n, h * strideH + kh * dilationH, w * strideW + kw * dilationW, c}, rewriter.getContext());
    // TFLite Depthwise Filter layout is typically [1, H, W, C]
    auto zero = rewriter.getAffineConstantExpr(0);
    auto filterMap = mlir::AffineMap::get(6, 0, {zero, kh, kw, c}, rewriter.getContext());
    auto outputMap = mlir::AffineMap::get(6, 0, {n, h, w, c}, rewriter.getContext());

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(6, mlir::utils::IteratorType::parallel);
    iteratorTypes[4] = mlir::utils::IteratorType::reduction; // kh
    iteratorTypes[5] = mlir::utils::IteratorType::reduction; // kw

    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, outputType, 
        mlir::ValueRange{input, filter}, 
        mlir::ValueRange{emptyTensor.getResult()},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, filterMap, outputMap},
        iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            b.create<mlir::linalg::YieldOp>(loc, args[2]);
        });

    rewriter.replaceOp(op, genericOp->getResults());
    return mlir::success();
  }
};

// Custom pass to orchestrate the TFL to Linalg conversion.
struct TFLToLinalgPass : public mlir::PassWrapper<TFLToLinalgPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFLToLinalgPass)

  void runOnOperation() override {
    auto module = getOperation();
    auto* context = &getContext();

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect, 
                          mlir::arith::ArithDialect, mlir::func::FuncDialect,
                          mlir::TFL::TFLDialect, mlir::quant::QuantDialect>();
    target.addIllegalOp<mlir::TFL::Conv2DOp, mlir::TFL::DepthwiseConv2DOp>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<Conv2DRewriter, DepthwiseConv2DRewriter>(context);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_file>" << std::endl;
    return 1;
  }

  mlir::MLIRContext context;
  context.loadDialect<mlir::TFL::TFLDialect, mlir::linalg::LinalgDialect,
                      mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect, mlir::quant::QuantDialect>();

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

  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<TFLToLinalgPass>());

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to lower to Linalg" << std::endl;
    return 1;
  }

  module->print(llvm::outs());
  return 0;
}

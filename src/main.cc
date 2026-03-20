#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/TilingDomain.h"

namespace {

// Helper to create tensor.empty for a given output type, handling dynamic batch.
mlir::Value createEmptyTensor(mlir::OpBuilder& builder, mlir::Location loc, 
                               mlir::RankedTensorType type, mlir::Value input) {
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (int i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i)) {
      dynamicSizes.push_back(builder.create<mlir::tensor::DimOp>(loc, input, i));
    }
  }
  return builder.create<mlir::tensor::EmptyOp>(loc, type.getShape(), type.getElementType(), dynamicSizes);
}

// Rewriter to convert tfl.conv_2d to linalg.generic (mimicking specialized prims).
struct Conv2DRewriter : public mlir::OpConversionPattern<mlir::TFL::Conv2DOp> {
  using OpConversionPattern<mlir::TFL::Conv2DOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::Conv2DOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!outputType) return mlir::failure();

    auto loc = op.getLoc();
    auto emptyTensor = createEmptyTensor(rewriter, loc, outputType, adaptor.getInput());

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
    iteratorTypes[4] = mlir::utils::IteratorType::reduction;
    iteratorTypes[5] = mlir::utils::IteratorType::reduction;
    iteratorTypes[6] = mlir::utils::IteratorType::reduction;

    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, outputType, 
        mlir::ValueRange{adaptor.getInput(), adaptor.getFilter()}, 
        mlir::ValueRange{emptyTensor},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, filterMap, outputMap},
        iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
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

    auto loc = op.getLoc();
    auto emptyTensor = createEmptyTensor(rewriter, loc, outputType, adaptor.getInput());

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
    auto zero = rewriter.getAffineConstantExpr(0);
    auto filterMap = mlir::AffineMap::get(6, 0, {zero, kh, kw, c}, rewriter.getContext());
    auto outputMap = mlir::AffineMap::get(6, 0, {n, h, w, c}, rewriter.getContext());

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(6, mlir::utils::IteratorType::parallel);
    iteratorTypes[4] = mlir::utils::IteratorType::reduction;
    iteratorTypes[5] = mlir::utils::IteratorType::reduction;

    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, outputType, 
        mlir::ValueRange{adaptor.getInput(), adaptor.getFilter()}, 
        mlir::ValueRange{emptyTensor},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, filterMap, outputMap},
        iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            b.create<mlir::linalg::YieldOp>(loc, args[2]);
        });

    rewriter.replaceOp(op, genericOp->getResults());
    return mlir::success();
  }
};

// Generic rewriter for all TFL constant-like operations using explicit name.
struct GenericConstantRewriter : public mlir::ConversionPattern {
  GenericConstantRewriter(mlir::MLIRContext *context, llvm::StringRef opName)
      : mlir::ConversionPattern(opName, 1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto valueAttr = op->getAttrOfType<mlir::ElementsAttr>("value");
    if (!valueAttr) return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, valueAttr);
    return mlir::success();
  }
};

// Rewriter for TFLite quantize/dequantize.
struct QuantRewriter : public mlir::ConversionPattern {
  QuantRewriter(mlir::MLIRContext *context, llvm::StringRef opName)
      : mlir::ConversionPattern(opName, 1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, op->getResultTypes(), operands);
    return mlir::success();
  }
};

struct TFLToLinalgPass : public mlir::PassWrapper<TFLToLinalgPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFLToLinalgPass)

  void runOnOperation() override {
    auto module = getOperation();
    auto* context = &getContext();

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect, 
                          mlir::arith::ArithDialect, mlir::func::FuncDialect,
                          mlir::quant::QuantDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    
    target.addIllegalDialect<mlir::TFL::TFLDialect>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<Conv2DRewriter, DepthwiseConv2DRewriter>(context);
    
    patterns.add<GenericConstantRewriter>(context, "tfl.pseudo_qconst");
    patterns.add<GenericConstantRewriter>(context, "tfl.pseudo_const");
    patterns.add<GenericConstantRewriter>(context, "tfl.qconst");
    patterns.add<GenericConstantRewriter>(context, "tfl.const");
    
    patterns.add<QuantRewriter>(context, "tfl.quantize");
    patterns.add<QuantRewriter>(context, "tfl.dequantize");

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Helper to get bitwidth of a type, including quantized types.
static int64_t getBitWidth(mlir::Type type) {
  if (type.isIntOrFloat()) return type.getIntOrFloatBitWidth();
  if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
    return qType.getStorageType().getIntOrFloatBitWidth();
  }
  return 0;
}

struct MemoryHierarchyAwareTilingPass : public mlir::PassWrapper<MemoryHierarchyAwareTilingPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryHierarchyAwareTilingPass)

  void runOnOperation() override {
    auto module = getOperation();
    const int64_t sram_limit_ = 256 * 1024; // 256 KB

    llvm::errs() << "Tuturu " << __LINE__ << "\n";

    llvm::SmallVector<mlir::TilingInterface> opsToWrap;
    module.walk([&](mlir::TilingInterface op) {
      opsToWrap.push_back(op);
    });

    for (auto op : opsToWrap) {
      // Replace op with a region that contains op
      mlir::Location loc = op->getLoc();
      mlir::OpBuilder builder(op);

      // Create the wrapper with original operands
      auto wrapperOp = builder.create<mlir::tiling_domain::TilingRegionOp>(
          loc, op->getResultTypes(), op->getOperands());

      mlir::Block* bodyBlock = builder.createBlock(&wrapperOp.getRegion());
      
      // Add block arguments to the block, matching the operand types
      for (auto operand : op->getOperands()) {
        bodyBlock->addArgument(operand.getType(), loc);
      }

      // Inside the region, the op should use block arguments instead of original operands
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        op->setOperand(i, bodyBlock->getArgument(i));
      }

      op->moveBefore(bodyBlock, bodyBlock->end());
      builder.setInsertionPointToEnd(bodyBlock);
      auto yieldOp = builder.create<mlir::tiling_domain::TilingYieldOp>(loc, op->getResults());
      
      // Replace all uses of the original op's results with the results of the wrapper op.
      // This MUST happen after the op is moved inside the region and its results are yielded.
      for (auto [oldRes, newRes] : llvm::zip(op->getResults(), wrapperOp.getResults())) {
        oldRes.replaceUsesWithIf(newRes, [&](mlir::OpOperand& operand) {
            return operand.getOwner() != yieldOp;
        });
      }
    }

    llvm::errs() << "Tuturu " << __LINE__ << "\n";
  }
};

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_file>" << std::endl;
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TFLDialect, mlir::linalg::LinalgDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::tensor::TensorDialect, mlir::quant::QuantDialect,
                  mlir::tiling_domain::TilingDomainDialect>();

  // Register TilingInterface external models for Linalg and Tensor.
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);

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

  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<TFLToLinalgPass>());
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to lower to Linalg" << std::endl;
    return 1;
  }

//   module->print(llvm::outs());

  mlir::PassManager pm2(&context);
  pm2.addPass(std::make_unique<MemoryHierarchyAwareTilingPass>());
  if (mlir::failed(pm2.run(*module))) {
    std::cerr << "Failed to run tiling" << std::endl;
    return 1;
  }

  module->print(llvm::outs());

  return 0;
}

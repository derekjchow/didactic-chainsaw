#include <algorithm>
#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/MemoryBuffer.h"

namespace {

// Rewrite tfl.conv_2d with stride S >= 2 (and stride_h == stride_w) into:
//
//   pad(input)  -> space_to_depth(input, S)
//   pad(filter) -> space_to_depth(filter, S)
//   conv_2d(stride=1, padding=VALID)
//
// The input is padded to account for SAME/VALID semantics and to make spatial
// dims divisible by S. The filter is padded so kH/kW become multiples of S,
// then space_to_depth folds the S*S spatial sub-blocks into the channel dim.
// ---------------------------------------------------------------------------
// Constant-folding helpers
// ---------------------------------------------------------------------------

// Extract a DenseElementsAttr from a defining op that is either a
// tfl.pseudo_qconst or an arith.constant.  Returns nullptr on failure.
static mlir::DenseElementsAttr getConstAttr(mlir::Value v) {
  auto* def = v.getDefiningOp();
  if (auto qc = mlir::dyn_cast_or_null<mlir::TFL::QConstOp>(def)) {
    return mlir::dyn_cast<mlir::DenseElementsAttr>(qc.getValue());
  }
  if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(def)) {
    return mlir::dyn_cast<mlir::DenseElementsAttr>(c.getValue());
  }
  return {};
}

static bool isQConst(mlir::Value v) {
  return mlir::isa_and_nonnull<mlir::TFL::QConstOp>(v.getDefiningOp());
}

// Create a new constant (QConstOp or arith::ConstantOp) from attributes.
static mlir::Value makeConst(mlir::PatternRewriter& rewriter, mlir::Location loc,
                             mlir::Type elemType,
                             llvm::ArrayRef<int64_t> shape,
                             llvm::ArrayRef<mlir::Attribute> values,
                             bool quantized) {
  mlir::RankedTensorType storageTy;
  if (auto qt = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
    storageTy = mlir::RankedTensorType::get(shape, qt.getStorageType());
  } else {
    storageTy = mlir::RankedTensorType::get(shape, elemType);
  }

  auto attr = mlir::DenseElementsAttr::get(storageTy, values);
  if (quantized) {
    auto qTy = mlir::RankedTensorType::get(shape, elemType);
    return rewriter.create<mlir::TFL::QConstOp>(
        loc, mlir::TypeAttr::get(qTy), attr);
  }
  return rewriter.create<mlir::arith::ConstantOp>(loc, attr);
}

// ---------------------------------------------------------------------------
// Fold tfl.pad(const, padding) - const
// ---------------------------------------------------------------------------
struct FoldConstPad : public mlir::OpRewritePattern<mlir::TFL::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::PadOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto inputAttr = getConstAttr(op.getInput());
    if (!inputAttr) {
      return mlir::failure();
    }

    // Padding amounts must be constant.
    mlir::DenseIntElementsAttr padAttr;
    if (auto c = op.getPadding().getDefiningOp<mlir::arith::ConstantOp>()) {
      padAttr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(c.getValue());
    } 
    if (!padAttr) {
      return mlir::failure();
    }

    auto inType = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto outType = mlir::cast<mlir::RankedTensorType>(op.getType());
    int rank = inType.getRank();
    auto inShape = inType.getShape();
    auto outShape = outType.getShape();

    auto pv = padAttr.getValues<int32_t>();
    llvm::SmallVector<int64_t> padBefore(rank);
    for (int i = 0; i < rank; ++i) {
      padBefore[i] = *(pv.begin() + i * 2);
    }

    auto elemType = inType.getElementType();

    int64_t totalIn = 1;
    for (auto d : inShape) {
      totalIn *= d;
    }
    int64_t totalOut = 1;
    for (auto d : outShape) {
      totalOut *= d;
    }

    auto inVals = inputAttr.getValues<mlir::Attribute>();
    // tfl.pad pads with zero in the storage type.
    auto zeroAttr = rewriter.getZeroAttr(inputAttr.getElementType());
    llvm::SmallVector<mlir::Attribute> outVals(totalOut, zeroAttr);

    // Copy each input element to its padded position.
    llvm::SmallVector<int64_t> coords(rank, 0);
    for (int64_t i = 0; i < totalIn; ++i) {
      int64_t outIdx = 0;
      for (int d = 0; d < rank; ++d) {
        outIdx = outIdx * outShape[d] + (coords[d] + padBefore[d]);
      }
      outVals[outIdx] = inVals[i];
      for (int d = rank - 1; d >= 0; --d) {
        if (++coords[d] < inShape[d]) {
          break;
        }
        coords[d] = 0;
      }
    }

    bool qc = isQConst(op.getInput());
    rewriter.replaceOp(
        op, makeConst(rewriter, op.getLoc(), elemType, outShape, outVals, qc));
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// Fold tfl.space_to_depth(const) - const
// ---------------------------------------------------------------------------
struct FoldConstSpaceToDepth
    : public mlir::OpRewritePattern<mlir::TFL::SpaceToDepthOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::SpaceToDepthOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto inputAttr = getConstAttr(op.getInput());
    if (!inputAttr) {
      return mlir::failure();
    }

    auto inType = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto outType = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (inType.getRank() != 4) {
      return mlir::failure();
    }

    int64_t S = op.getBlockSize();
    auto inShape = inType.getShape();
    int64_t N = inShape[0], H = inShape[1], W = inShape[2], C = inShape[3];

    auto elemType = inType.getElementType();

    auto inVals = inputAttr.getValues<mlir::Attribute>();

    auto outShape = outType.getShape();
    int64_t oH = H / S, oW = W / S, oC = C * S * S;
    int64_t totalOut = N * oH * oW * oC;
    llvm::SmallVector<mlir::Attribute> outVals(totalOut);

    // input[n, h, w, c] -> output[n, h/S, w/S, c*S*S + (h%S)*S + (w%S)]
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          for (int64_t c = 0; c < C; ++c) {
            int64_t inIdx = ((n * H + h) * W + w) * C + c;
            int64_t outIdx =
                ((n * oH + h / S) * oW + w / S) * oC +
                c * S * S + (h % S) * S + (w % S);
            outVals[outIdx] = inVals[inIdx];
          }
        }
      }
    }

    bool qc = isQConst(op.getInput());
    rewriter.replaceOp(
        op, makeConst(rewriter, op.getLoc(), elemType, outShape, outVals, qc));
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// Pad a 4-D tensor along the spatial (H, W) dimensions.  Returns the input
// unchanged when all padding amounts are zero.  When pad_zero_point is true
// and the element type is quantized, the pad value is the quantization
// zero-point (via PadV2); otherwise raw-0 padding is used (via Pad).
// ---------------------------------------------------------------------------
static mlir::Value padSpatial(mlir::PatternRewriter& rewriter,
                              mlir::Location loc, mlir::Value input,
                              int64_t pad_top, int64_t pad_bottom,
                              int64_t pad_left, int64_t pad_right,
                              bool pad_zero_point = false) {
  if (!pad_top && !pad_bottom && !pad_left && !pad_right)
    return input;

  auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto shape = inType.getShape();
  auto elemType = inType.getElementType();

  llvm::SmallVector<int32_t, 8> pv = {
      0, 0,
      static_cast<int32_t>(pad_top), static_cast<int32_t>(pad_bottom),
      static_cast<int32_t>(pad_left), static_cast<int32_t>(pad_right),
      0, 0};
  auto padTy = mlir::RankedTensorType::get({4, 2}, rewriter.getI32Type());
  auto padConst = rewriter.create<mlir::arith::ConstantOp>(
      loc, mlir::DenseIntElementsAttr::get(padTy, pv));

  auto paddedTy = mlir::RankedTensorType::get(
      {shape[0], shape[1] + pad_top + pad_bottom,
       shape[2] + pad_left + pad_right, shape[3]},
      elemType);

  if (pad_zero_point) {
    if (auto qtype =
          mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
      auto scalarQTy = mlir::RankedTensorType::get({}, elemType);
      auto storageTy = mlir::RankedTensorType::get({}, qtype.getStorageType());
      auto zpAttr = mlir::DenseIntElementsAttr::get(
          storageTy, static_cast<int8_t>(qtype.getZeroPoint()));
      auto zpVal = rewriter.create<mlir::TFL::QConstOp>(
          loc, mlir::TypeAttr::get(scalarQTy), zpAttr);
      return rewriter.create<mlir::TFL::PadV2Op>(loc, paddedTy, input,
                                                  padConst, zpVal);
    }
  }
  return rewriter.create<mlir::TFL::PadOp>(loc, paddedTy, input, padConst);
}

// ---------------------------------------------------------------------------
// Rewrite strided conv - pad + space_to_depth + stride-1 conv
// ---------------------------------------------------------------------------
struct SpaceToDepthConvRewriter
    : public mlir::OpRewritePattern<mlir::TFL::Conv2DOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::Conv2DOp op,
      mlir::PatternRewriter& rewriter) const override {
    int64_t S = op.getStrideH();
    if (S < 2 || op.getStrideW() != S) return mlir::failure();
    if (op.getDilationHFactor() != 1 || op.getDilationWFactor() != 1)
      return mlir::failure();

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    auto filterType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getFilter().getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!inputType || !filterType || !outputType) return mlir::failure();
    if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return mlir::failure();

    auto loc = op.getLoc();

    // Input [N, H, W, C].
    int64_t N = inputType.getDimSize(0);
    int64_t H = inputType.getDimSize(1);
    int64_t W = inputType.getDimSize(2);
    int64_t C = inputType.getDimSize(3);

    // Filter [F, kH, kW, C_in].
    int64_t F = filterType.getDimSize(0);
    int64_t kH = filterType.getDimSize(1);
    int64_t kW = filterType.getDimSize(2);

    // Output spatial dims from original padding mode.
    auto paddingStr =
        op->getAttrOfType<mlir::StringAttr>("padding").getValue();
    bool isSame = (paddingStr == "SAME");
    int64_t oH = isSame ? (H + S - 1) / S : (H - kH) / S + 1;
    int64_t oW = isSame ? (W + S - 1) / S : (W - kW) / S + 1;

    // New filter spatial dims (ceil kH/kW to next multiple of S).
    int64_t kH_new = (kH + S - 1) / S;
    int64_t kW_new = (kW + S - 1) / S;
    int64_t kH_pad = kH_new * S;
    int64_t kW_pad = kW_new * S;

    // Padded input dims (must be divisible by S).
    int64_t H_pad = S * (oH + kH_new - 1);
    int64_t W_pad = S * (oW + kW_new - 1);

    // Per-side input padding. For SAME we replicate TFL's symmetric split;
    // for VALID all padding goes to bottom/right.
    int64_t pad_top = 0, pad_left = 0;
    if (isSame) {
      int64_t total_h = std::max<int64_t>(0, (oH - 1) * S + kH - H);
      int64_t total_w = std::max<int64_t>(0, (oW - 1) * S + kW - W);
      pad_top = total_h / 2;
      pad_left = total_w / 2;
    }
    int64_t pad_bottom = H_pad - H - pad_top;
    int64_t pad_right = W_pad - W - pad_left;

    auto inputElem = inputType.getElementType();
    auto filterElem = filterType.getElementType();

    // --- Pad input ---
    mlir::Value input = padSpatial(rewriter, loc, op.getInput(),
                                   pad_top, pad_bottom, pad_left, pad_right,
                                   /*pad_zero_point=*/true);

    // --- Space-to-depth on input ---
    auto s2dInTy = mlir::RankedTensorType::get(
        {N, H_pad / S, W_pad / S, C * S * S}, inputElem);
    auto s2dIn = rewriter.create<mlir::TFL::SpaceToDepthOp>(
        loc, s2dInTy, input, rewriter.getI32IntegerAttr(S));

    // --- Pad filter ---
    mlir::Value filter = padSpatial(rewriter, loc, op.getFilter(),
                                    0, kH_pad - kH, 0, kW_pad - kW);

    // --- Space-to-depth on filter ---
    auto s2dFiltTy = mlir::RankedTensorType::get(
        {F, kH_new, kW_new, C * S * S}, filterElem);
    auto s2dFilt = rewriter.create<mlir::TFL::SpaceToDepthOp>(
        loc, s2dFiltTy, filter, rewriter.getI32IntegerAttr(S));

    // --- New Conv2D: stride 1, VALID padding ---
    // Build via OperationState to avoid depending on the exact ODS builder
    // parameter order for TFL::Conv2DOp.
    mlir::OperationState convState(
        loc, mlir::TFL::Conv2DOp::getOperationName());
    convState.addOperands({s2dIn, s2dFilt, op.getBias()});
    convState.addTypes({outputType});
    // Copy every attribute from the original conv, overriding stride & padding.
    for (auto attr : op->getAttrs()) {
      llvm::StringRef name = attr.getName().strref();
      if (name == "stride_h" || name == "stride_w")
        convState.addAttribute(name, rewriter.getI32IntegerAttr(1));
      else if (name == "padding")
        convState.addAttribute(name, rewriter.getStringAttr("VALID"));
      else
        convState.addAttribute(name, attr.getValue());
    }
    auto* newConv = rewriter.create(convState);
    rewriter.replaceOp(op, newConv->getResults());
    return mlir::success();
  }
};

struct SpaceToDepthConvPass
    : public mlir::PassWrapper<SpaceToDepthConvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpaceToDepthConvPass)

  llvm::StringRef getArgument() const override {
    return "tfl-space-to-depth-conv";
  }
  llvm::StringRef getDescription() const override {
    return "Rewrite strided convolutions via pad + space_to_depth + stride-1 "
           "conv.";
  }

  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<SpaceToDepthConvRewriter, FoldConstPad,
                 FoldConstSpaceToDepth>(ctx);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tflite_file> [output.tflite]"
              << std::endl;
    return 1;
  }
  std::string output_path =
      argc >= 3 ? argv[2] : "transformed.tflite";

  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TFLDialect, mlir::arith::ArithDialect>();

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
  pm.addPass(std::make_unique<SpaceToDepthConvPass>());
  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Space-to-depth conv pass failed" << std::endl;
    return 1;
  }

  tflite::FlatbufferExportOptions options;
  std::string serialized_flatbuffer;
  if (!tflite::MlirToFlatBufferTranslateFunction(
          *module, options, &serialized_flatbuffer)) {
    std::cerr << "Failed to export MLIR to FlatBuffer" << std::endl;
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream out_file(output_path, ec);
  if (ec) {
    std::cerr << "Failed to open output file: " << ec.message() << std::endl;
    return 1;
  }
  out_file.write(serialized_flatbuffer.data(), serialized_flatbuffer.size());

  return 0;
}

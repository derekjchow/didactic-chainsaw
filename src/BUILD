cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:tfl_stablehlo",

        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:linalg_passes",

        #"@org_tensorflow//tensorflow/compiler/mlir/tosa:tfl_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        # "@llvm-project//mlir:QuantDialect",
        # "@llvm-project//mlir:TosaDialect",
        # "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//llvm:Support",
        "@com_google_absl//absl/strings",
    ],    
)
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

gentbl_cc_library(
    name = "tiling_domain_inc_gen",
    # compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "TilingDomain.h.inc": ["-gen-dialect-decls"],
        "TilingDomain.cpp.inc": ["-gen-dialect-defs"],
        "TilingDomainOps.h.inc": ["-gen-op-decls"],
        "TilingDomainOps.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TilingDomain.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "tiling_domain",
    hdrs = ["TilingDomain.h"],
    srcs = ["TilingDomain.cc"],
    deps = [
        ":tiling_domain_inc_gen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

cc_binary(
    name = "spacial_conv",
    srcs = ["spacial_conv.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//llvm:Support",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":tiling_domain",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/stablehlo:tfl_stablehlo",

        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:linalg_passes",


        # "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        # "@llvm-project//mlir:RegisterAllExtensions",  # buildcleaner: keep


        #"@org_tensorflow//tensorflow/compiler/mlir/tosa:tfl_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:TensorTilingInterfaceImpl",
        "@llvm-project//mlir:TilingInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//llvm:Support",
        "@com_google_absl//absl/strings",
    ],    
)
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")
load("//larq_compute_engine/tflite:build_defs.bzl", "ruy_copts_base")

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)


cc_library(
    name = "ternary_op",
    srcs = ["ternary.cc"],
    deps = [
        ":utils",
        ":lce_op_kernels",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    visibility = ["//visibility:public"],  # Adjust visibility as needed
)

pybind_extension(
    name = "ternary_op_py",
    srcs = ["ternary.cc"],
    linkopts = tflite_linkopts(),
    module_name = "ternary_op_mod",
    deps = [
        ":lce_op_kernels",
        "//larq_compute_engine/tflite/python:interpreter_wrapper_utils",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@pybind11",
    ],
)


cc_library(
    name = "lce_op_kernels",
    srcs = [
        "bconv2d.cc",
        "bmaxpool.cc",
        "quantization.cc",
        "ternary.cc",
        "unpack_ternary.cc",
    ],
    hdrs = [
        "lce_ops_register.h",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    deps = [
        ":utils",
        "//larq_compute_engine/core:bmaxpool",
        "//larq_compute_engine/core/bconv2d:optimized_bgemm",
        "//larq_compute_engine/core/bconv2d:optimized_indirect_bgemm",
        "//larq_compute_engine/core/bconv2d:params",
        "//larq_compute_engine/core/bconv2d:reference",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "//larq_compute_engine/core/bitpacking:utils",
        "//larq_compute_engine/core/indirect_bgemm:kernels",
        "//larq_compute_engine/core/ternary",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:type_to_tflitetype",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/tools:logging",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@ruy//ruy/profiler:instrumentation",
    ],
    alwayslink = 1,
)

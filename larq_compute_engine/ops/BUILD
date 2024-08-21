load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension", "tf_custom_op_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load("@org_tensorflow//tensorflow:tensorflow.bzl","tf_copts")
load("@pypi//:requirements.bzl", tf_requirement = "requirement")

package(default_visibility = ["//visibility:public"])


tf_custom_op_library(
    name = "custom_ops.so",
    srcs = ["custom_ops.cc"],
    linkopts = tf_copts() + ["-Lvenv/lib64/python3.11/site-packages/tensorflow -l:libtensorflow_framework.so.2"],
    deps = [
        "//larq_compute_engine/core/ternary:ternary",
        # "@org_tensorflow//tensorflow/core:framework",
        # "@org_tensorflow//tensorflow/core:ops",
        # "@org_tensorflow//tensorflow/c:c_api_experimental",
    ],
)

# cc_binary(
#     name = "custom_ops.so",
#     srcs = ["custom_ops.cc"],
#     linkopts = tf_copts() + ["-L/home/crulis/Documents/projects/compute-engine/venv/lib64/python3.11/site-packages/tensorflow -l:libtensorflow_framework.so.2"],
#     linkshared = 1,
#     deps = [
#         "@org_tensorflow//tensorflow/core:framework",
#         "@org_tensorflow//tensorflow/core:lib",
#         # "@org_tensorflow//tensorflow:libtensorflow_framework",
#         # "@org_tensorflow//tensorflow:tf_header_lib",
#         # "@org_tensorflow//tensorflow:libtensorflow_framework.so"
#         # "@pybind11",
#     ]
# )

# pybind_extension(
#     name = "custom_ops_py_wrapper",
#     srcs = ["custom_ops.cc"],  # Or the wrapper file if you have a separate one
#     module_name = "custom_ops",
#     linkopts = tflite_linkopts(),
#     deps = [
#         ":custom_ops.so",
#         "@pybind11",
#     ],
# )


# pybind_extension(
#     name = "custom_ops_tmp",
#     srcs = ["custom_ops.cc"],
#     linkopts = tflite_linkopts(),
#     module_name = "custom_ops",
#     deps = [
#         ":custom_ops.so",
#         "@org_tensorflow//tensorflow/lite/c:common",
#         "@org_tensorflow//tensorflow/lite:framework",
#         "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
#         "//larq_compute_engine/tflite/kernels:lce_op_kernels",
#         "@pybind11",
#     ],
# )

pybind_extension(
    name = "custom_ops_tmp",
    srcs = ["custom_ops.cc"],
    module_name = "custom_ops",
    linkopts = tf_copts() + ["-L/home/crulis/Documents/projects/compute-engine/venv/lib64/python3.11/site-packages/tensorflow -l:libtensorflow_framework.so.2"],
    deps = [
        ":custom_ops.so",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@pybind11",
    ],
)

py_library(
    name = "custom_ops_py",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":custom_ops_tmp",
        tf_requirement("numpy"),
    ],
)
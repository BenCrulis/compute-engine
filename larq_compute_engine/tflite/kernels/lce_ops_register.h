#ifndef COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_
#define COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/tools/logging.h"

// This file contains forward declaration of all custom ops
// implemented in LCE which can be used to link against LCE library.

namespace compute_engine {
namespace tflite {

using namespace ::tflite;

TfLiteRegistration* Register_QUANTIZE();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_BCONV_2D();
TfLiteRegistration* Register_BCONV_2D_REF();
TfLiteRegistration* Register_BCONV_2D_OPT_INDIRECT_BGEMM();
TfLiteRegistration* Register_BMAXPOOL_2D();
TfLiteRegistration* Register_TERNARY_OPT();
TfLiteRegistration* Register_UNPACK_TERNARY_OPT();


void register_all_custom_ops(MutableOpResolver* resolver) {
  resolver->AddCustom("TernaryMatmul",
                      compute_engine::tflite::Register_TERNARY_OPT());
  resolver->AddCustom("UnpackTernary",
                      compute_engine::tflite::Register_UNPACK_TERNARY_OPT());
};


void register_all_ops(uintptr_t mutable_op_resolver_ptr) {
  auto* resolver = reinterpret_cast<MutableOpResolver*>(mutable_op_resolver_ptr);

  resolver->AddCustom("LceQuantize",
                      compute_engine::tflite::Register_QUANTIZE());
  resolver->AddCustom("LceDequantize",
                      compute_engine::tflite::Register_DEQUANTIZE());
  resolver->AddCustom("LceBconv2d",
                      compute_engine::tflite::Register_BCONV_2D());
    
  resolver->AddCustom("LceBMaxPool2d",
                      compute_engine::tflite::Register_BMAXPOOL_2D());

  // custom operations
  register_all_custom_ops(resolver);
};

// By calling this function on TF lite mutable op resolver, all LCE custom ops
// will be registerd to the op resolver.
inline void RegisterLCECustomOps(::tflite::MutableOpResolver* resolver,
                                 const bool use_reference_bconv = false,
                                 const bool use_indirect_bgemm = false) {
  if (use_reference_bconv && use_indirect_bgemm) {
    TFLITE_LOG(WARN)
        << "WARNING: 'use_reference_bconv' and `use_indirect_bgemm` "
           "are both set to true. use_indirect_bgemm==true "
           "will have no effect.";
  }
  resolver->AddCustom("LceQuantize",
                      compute_engine::tflite::Register_QUANTIZE());
  resolver->AddCustom("LceDequantize",
                      compute_engine::tflite::Register_DEQUANTIZE());
  if (use_reference_bconv) {
    resolver->AddCustom("LceBconv2d",
                        compute_engine::tflite::Register_BCONV_2D_REF());
  } else {
    if (use_indirect_bgemm) {
      resolver->AddCustom(
          "LceBconv2d",
          compute_engine::tflite::Register_BCONV_2D_OPT_INDIRECT_BGEMM());
    } else {
      resolver->AddCustom("LceBconv2d",
                          compute_engine::tflite::Register_BCONV_2D());
    }
  }
  resolver->AddCustom("LceBMaxPool2d",
                      compute_engine::tflite::Register_BMAXPOOL_2D());

  // custom operations
  register_all_custom_ops(resolver);
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_

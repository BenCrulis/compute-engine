#include <cmath>
#include <cstdint>

//#include "tensorflow/core/framework/op_def.pb.h"

#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

#include "tensorflow/core/framework/op.h" // for registration of custom op
#include "tensorflow/core/framework/shape_inference.h"

#include "larq_compute_engine/core/ternary/ternary.h"


using namespace tflite;
namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace unpacking {

using ce::core::TBitpacked;

struct OpData {
    
};

void* Init(TfLiteContext* context, const char* buffer, std::size_t length) {
  // printf("initializing TFLite Ternary op\n");
  auto* op_data = new OpData{};
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {

  const auto* weights = GetInput(context, node, 0);
  const auto* size = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, weights->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, size->type, kTfLiteInt32);

  TF_LITE_ENSURE(context, IsConstantOrPersistentTensor(size));

  RuntimeShape weights_shape = GetTensorShape(weights);

  int size_data = GetTensorData<int>(size)[0];
  

  TfLiteIntArray* shape = TfLiteIntArrayCreate(2);
  shape->data[0] = weights_shape.Dims(0);
  shape->data[1] = size_data;

  context->ResizeTensor(context, output, shape);

  // printf("returning OK for prepare\n");
  return kTfLiteOk;
}


TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* weights = GetInput(context, node, 0);
  auto* size = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  RuntimeShape weights_shape = GetTensorShape(weights);
  RuntimeShape output_shape = GetTensorShape(output);

  const uint8* weight_data = GetTensorData<uint8>(weights);
  float* output_data = GetTensorData<float>(output);
  int size_data = GetTensorData<int>(size)[0];
  
  int chan_out = weights_shape.Dims(0);

  // unpack_ternary(weight_data, output_data, size_data, chan_out);
  unpack_ternary_threaded(weight_data, output_data, size_data, chan_out);

  TfLiteIntArray* shape = TfLiteIntArrayCreate(2);

  shape->data[0] = chan_out;
  shape->data[1] = size_data;

  context->ResizeTensor(context, output, shape);

  // TfLiteIntArrayFree(shape);

  return kTfLiteOk;
}

} // namespace unpacking

TfLiteRegistration* Register_UNPACK_TERNARY_OPT() {
  // printf("registering Unpack Ternary\n");
  static TfLiteRegistration r = {
      unpacking::Init,
      unpacking::Free,
      unpacking::Prepare,
      unpacking::Eval};
  return &r;
}


}  // namespace tflite
}  // namespace compute_engine

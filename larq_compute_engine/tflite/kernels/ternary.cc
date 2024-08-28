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

#include <thread>

#include <emmintrin.h>
#include <immintrin.h>


using namespace tflite;
namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace ternary {

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

  const auto* input = GetInput(context, node, 0);
  const auto* weights = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, weights->type, kTfLiteUInt8);

  int num_dim = NumDimensions(input);

  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape weights_shape = GetTensorShape(weights);

  int input_channels = input_shape.Dims(num_dim-1);

  // the second dimension of the weight tensor is compressed, check that uncompressed size matches with input channel size
  int compressed_chan_size = input_channels / 4;
  int remainder = input_channels % 4;
  if (remainder != 0) {
    compressed_chan_size += 1;
  }

  TF_LITE_ENSURE_EQ(context, compressed_chan_size, weights_shape.Dims(weights_shape.DimensionsCount() - 1));

  // Determine the output dimensions and allocate the output buffer
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dim);

  // copy the first dimensions to the input
  for (int i = 0; i < num_dim - 1 ; i++) {
    output_shape->data[i] = SizeOfDimension(input, i);
  }

  // compute the last dimension using the weight shape
  output_shape->data[num_dim - 1] = SizeOfDimension(weights, 0);

  // printf("resizing tensor\n");
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // printf("returning OK for prepare\n");
  return kTfLiteOk;
}


static inline void accum_ternary(float* out, float input, uint8 w) {
  uint mw = w & 0b00000011;
  if (mw == 0) {
    // printf("subtracting %f\n", input);
    *out -= input;
  }
  else if (mw == 2) {
    // printf("adding %f\n", input);
    *out += input;
  }
}


// #define accum_ternary_macro(out, input, w) { \
//   const uint8 mw = w & 0b00000011;                  \
//   if (mw == 0) {                             \
//     out -= input;                            \
//   }                                          \
//   else if (mw == 2) {                        \
//     out += input;                            \
//   }                                          \
// }

#define accum_ternary_macro(out, input, w) { \
  const uint8 mw = w & 0b00000011;           \
  float val = 0.0;                           \
  if (mw == 0) {                             \
    val = -input;                            \
  }                                          \
  else if (mw == 2) {                        \
    val = input;                             \
  }                                          \
  out += val;                                \
}


#define accum_ternary_macro_mask(out, input, w, mask, val_neg_one, val_pos_one) { \
  const uint8 mw = w & mask;                 \
  float val = 0.0;                           \
  if (mw == val_neg_one) {                   \
    val = -input;                            \
  }                                          \
  else if (mw == val_pos_one) {              \
    val = input;                             \
  }                                          \
  out += val;                                \
}


void reference_implementation(const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data) {
  const int remainder = chan_in % 4;

  const int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;


  for (int i = 0 ; i < batch_size ; i++) {
    for (int j = 0 ; j < chan_out ; j++) {
      // printf("computing i=%i, j=%i\n", i, j);
      const int out_idx = i*chan_out+j;
      float out0 = 0.0; float out1 = 0.0; float out2 = 0.0; float out3 = 0.0;
      for (int k = 0; k < compressed_chan_size_target; k++) {
        uint8 w = weight_data[j*compressed_chan_size + k];
        int in_idx_base = i*chan_in + k*4;
        // printf("k=%i, w=%i\n", k, w);
        accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
        accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
        accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
        accum_ternary_macro(out3, input_data[in_idx_base+3], w);
      }
      uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
      int in_idx_base = i*chan_in + chan_in - 5;
      if (remainder > 0) {
        accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
      }
      if (remainder > 1) {
        accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
      }
      if (remainder > 2) {
        accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
      }
      output_data[out_idx] = out0 + out1 + out2 + out3;
    }
  }
}


const long int BLOCK_SIZE = 4;
void tiled_implementation(const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data) {
  // WIP
  const int remainder = chan_in % 4;

  const int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;


  for (int i = 0 ; i < batch_size ; i++) {
    for (int j = 0; j < chan_out; j++) {
      output_data[i*chan_out + j] = 0.0;
    }

    // tiling loop
    for (int kk = 0 ; kk < chan_out ; kk += BLOCK_SIZE) {

      for (int j = 0; j < chan_out; j++) {
        const int out_idx = i*chan_out+j;
        float out0 = 0.0; float out1 = 0.0; float out2 = 0.0; float out3 = 0.0;
        for (int k = kk; k < compressed_chan_size_target && k < kk + BLOCK_SIZE; k++) {
          uint8 w = weight_data[j*compressed_chan_size + k];
          int in_idx_base = i*chan_in + k*4;
          accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
          accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
          accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
          accum_ternary_macro(out3, input_data[in_idx_base+3], w);
        }
        uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
        int in_idx_base = i*chan_in + chan_in - 5;
        if (remainder > 0) {
          accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
        }
        if (remainder > 1) {
          accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
        }
        if (remainder > 2) {
          accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
        }
        output_data[out_idx] += out0 + out1 + out2 + out3;
      } // output loop

    } // tile loop

  } // batch loop

}


void tiled_implementation2(const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data) {
  // WIP
  const int remainder = chan_in % 4;

  const int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;

  // tiling loop
  for (int kk = 0 ; kk < chan_out ; kk += BLOCK_SIZE) {

    for (int i = 0 ; i < batch_size ; i++) {

      for (int j = kk; j < chan_out && j < kk + BLOCK_SIZE; j++) {
        const int out_idx = i*chan_out+j;
        float out0 = 0.0; float out1 = 0.0; float out2 = 0.0; float out3 = 0.0;
        for (int k = 0; k < compressed_chan_size_target; k++) {
          uint8 w = weight_data[j*compressed_chan_size + k];
          int in_idx_base = i*chan_in + k*4;
          accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
          accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
          accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
          accum_ternary_macro(out3, input_data[in_idx_base+3], w);
        }
        uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
        int in_idx_base = i*chan_in + chan_in - 5;
        if (remainder > 0) {
          accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
        }
        if (remainder > 1) {
          accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
        }
        if (remainder > 2) {
          accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
        }
        output_data[out_idx] = out0 + out1 + out2 + out3;
      } // output loop

    } // batch loop

  } // tile loop

}


void switch_increment(const uint8 w, const float* input, float* out) {
  switch (w) {
    case 0b00000000: *out -=  input[0] + input[1] + input[2] + input[3]; break;
    case 0b00000010: *out += -input[0] - input[1] - input[2] + input[3]; break;
    case 0b00001000: *out += -input[0] - input[1] + input[2] - input[3]; break;
    case 0b00100000: *out += -input[0] + input[1] - input[2] - input[3]; break;
    case 0b10000000: *out +=  input[0] - input[1] - input[2] - input[3]; break;
    case 0b00001010: *out += -input[0] - input[1] + input[2] + input[3]; break;
    case 0b00100010: *out += -input[0] + input[1] - input[2] + input[3]; break;
    case 0b10000010: *out +=  input[0] - input[1] - input[2] + input[3]; break;
    case 0b00101010: *out += -input[0] + input[1] + input[2] + input[3]; break;
    case 0b10001010: *out +=  input[0] - input[1] + input[2] + input[3]; break;
    case 0b10101010: *out +=  input[0] + input[1] + input[2] + input[3]; break;
    case 0b10100010: *out +=  input[0] + input[1] - input[2] + input[3]; break;
    case 0b00101000: *out += -input[0] + input[1] + input[2] - input[3]; break;
    case 0b10001000: *out +=  input[0] - input[1] + input[2] - input[3]; break;
    case 0b10101000: *out +=  input[0] + input[1] + input[2] - input[3]; break;
    case 0b10100000: *out +=  input[0] + input[1] - input[2] - input[3]; break;
    case 0b00000001: *out += -input[0] - input[1] - input[2]           ; break;
    case 0b00001001: *out += -input[0] - input[1] + input[2]           ; break;
    case 0b00100001: *out += -input[0] + input[1] - input[2]           ; break;
    case 0b10000001: *out +=  input[0] - input[1] - input[2]           ; break;
    case 0b00101001: *out += -input[0] + input[1] + input[2]           ; break;
    case 0b10001001: *out +=  input[0] - input[1] + input[2]           ; break;
    case 0b10100001: *out +=  input[0] + input[1] - input[2]           ; break;
    case 0b10101001: *out +=  input[0] + input[1] + input[2]           ; break;
    case 0b00000100: *out += -input[0] - input[1]            - input[3]; break;
    case 0b00000110: *out += -input[0] - input[1]            + input[3]; break;
    case 0b00100100: *out += -input[0] + input[1]            - input[3]; break;
    case 0b10000100: *out +=  input[0] - input[1]            - input[3]; break;
    case 0b00100110: *out += -input[0] + input[1]            + input[3]; break;
    case 0b10000110: *out +=  input[0] - input[1]            + input[3]; break;
    case 0b10100100: *out +=  input[0] + input[1]            - input[3]; break;
    case 0b10100110: *out +=  input[0] + input[1]            + input[3]; break;
    case 0b00010000: *out += -input[0]            - input[2] - input[3]; break;
    case 0b00010010: *out += -input[0]            - input[2] + input[3]; break;
    case 0b00011000: *out += -input[0]            + input[2] - input[3]; break;
    case 0b10010000: *out +=  input[0]            - input[2] - input[3]; break;
    case 0b00011010: *out += -input[0]            + input[2] + input[3]; break;
    case 0b10010010: *out +=  input[0]            - input[2] + input[3]; break;
    case 0b10011000: *out +=  input[0]            + input[2] - input[3]; break;
    case 0b10011010: *out +=  input[0]            + input[2] + input[3]; break;
    case 0b01000000: *out +=           - input[1] - input[2] - input[3]; break;
    case 0b01000010: *out +=           - input[1] - input[2] + input[3]; break;
    case 0b01001000: *out +=           - input[1] + input[2] - input[3]; break;
    case 0b01100000: *out +=           + input[1] - input[2] - input[3]; break;
    case 0b01001010: *out +=           - input[1] + input[2] + input[3]; break;
    case 0b01100010: *out +=           + input[1] - input[2] + input[3]; break;
    case 0b01101000: *out +=           + input[1] + input[2] - input[3]; break;
    case 0b01101010: *out +=           + input[1] + input[2] + input[3]; break;
    case 0b00000101: *out += -input[0] - input[1]                      ; break;
    case 0b00100101: *out += -input[0] + input[1]                      ; break;
    case 0b10000101: *out +=  input[0] - input[1]                      ; break;
    case 0b10100101: *out +=  input[0] + input[1]                      ; break;
    case 0b00010001: *out += -input[0]            - input[2]           ; break;
    case 0b00011001: *out += -input[0]            + input[2]           ; break;
    case 0b10010001: *out +=  input[0]            - input[2]           ; break;
    case 0b10011001: *out +=  input[0]            + input[2]           ; break;
    case 0b01000001: *out +=           - input[1] - input[2]           ; break;
    case 0b01001001: *out +=           - input[1] + input[2]           ; break;
    case 0b01100001: *out +=             input[1] - input[2]           ; break;
    case 0b01101001: *out +=             input[1] + input[2]           ; break;
    case 0b00010100: *out += -input[0]                       - input[3]; break;
    case 0b00010110: *out += -input[0]                       + input[3]; break;
    case 0b10010100: *out +=  input[0]                       - input[3]; break;
    case 0b10010110: *out +=  input[0]                       + input[3]; break;
    case 0b01000100: *out +=           - input[1]            - input[3]; break;
    case 0b01000110: *out +=           - input[1]            + input[3]; break;
    case 0b01100100: *out +=             input[1]            - input[3]; break;
    case 0b01100110: *out +=             input[1]            + input[3]; break;
    case 0b01010000: *out +=                      - input[2] - input[3]; break;
    case 0b01010010: *out +=                      - input[2] + input[3]; break;
    case 0b01011000: *out +=                        input[2] - input[3]; break;
    case 0b01011010: *out +=                        input[2] + input[3]; break;
    case 0b00010101: *out += -input[0]                                 ; break;
    case 0b10010101: *out +=  input[0]                                 ; break;
    case 0b01000101: *out +=           - input[1]                      ; break;
    case 0b01100101: *out +=             input[1]                      ; break;
    case 0b01010001: *out +=                      - input[2]           ; break;
    case 0b01011001: *out +=                        input[2]           ; break;
    case 0b01010100: *out +=                                 - input[3]; break;
    case 0b01010110: *out +=                                   input[3]; break;
    default: break;
  }
}


const __m128 unpack_table[256] = {
  _mm_set_ps(-1, -1, -1, -1),
  _mm_set_ps(0, -1, -1, -1),
  _mm_set_ps(1, -1, -1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, -1, -1),
  _mm_set_ps(0, 0, -1, -1),
  _mm_set_ps(1, 0, -1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, -1, -1),
  _mm_set_ps(0, 1, -1, -1),
  _mm_set_ps(1, 1, -1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 0, -1),
  _mm_set_ps(0, -1, 0, -1),
  _mm_set_ps(1, -1, 0, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 0, -1),
  _mm_set_ps(0, 0, 0, -1),
  _mm_set_ps(1, 0, 0, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 0, -1),
  _mm_set_ps(0, 1, 0, -1),
  _mm_set_ps(1, 1, 0, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 1, -1),
  _mm_set_ps(0, -1, 1, -1),
  _mm_set_ps(1, -1, 1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 1, -1),
  _mm_set_ps(0, 0, 1, -1),
  _mm_set_ps(1, 0, 1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 1, -1),
  _mm_set_ps(0, 1, 1, -1),
  _mm_set_ps(1, 1, 1, -1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, -1, 0),
  _mm_set_ps(0, -1, -1, 0),
  _mm_set_ps(1, -1, -1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, -1, 0),
  _mm_set_ps(0, 0, -1, 0),
  _mm_set_ps(1, 0, -1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, -1, 0),
  _mm_set_ps(0, 1, -1, 0),
  _mm_set_ps(1, 1, -1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 0, 0),
  _mm_set_ps(0, -1, 0, 0),
  _mm_set_ps(1, -1, 0, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 0, 0),
  _mm_set_ps(0, 0, 0, 0),
  _mm_set_ps(1, 0, 0, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 0, 0),
  _mm_set_ps(0, 1, 0, 0),
  _mm_set_ps(1, 1, 0, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 1, 0),
  _mm_set_ps(0, -1, 1, 0),
  _mm_set_ps(1, -1, 1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 1, 0),
  _mm_set_ps(0, 0, 1, 0),
  _mm_set_ps(1, 0, 1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 1, 0),
  _mm_set_ps(0, 1, 1, 0),
  _mm_set_ps(1, 1, 1, 0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, -1, 1),
  _mm_set_ps(0, -1, -1, 1),
  _mm_set_ps(1, -1, -1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, -1, 1),
  _mm_set_ps(0, 0, -1, 1),
  _mm_set_ps(1, 0, -1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, -1, 1),
  _mm_set_ps(0, 1, -1, 1),
  _mm_set_ps(1, 1, -1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 0, 1),
  _mm_set_ps(0, -1, 0, 1),
  _mm_set_ps(1, -1, 0, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 0, 1),
  _mm_set_ps(0, 0, 0, 1),
  _mm_set_ps(1, 0, 0, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 0, 1),
  _mm_set_ps(0, 1, 0, 1),
  _mm_set_ps(1, 1, 0, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, -1, 1, 1),
  _mm_set_ps(0, -1, 1, 1),
  _mm_set_ps(1, -1, 1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 0, 1, 1),
  _mm_set_ps(0, 0, 1, 1),
  _mm_set_ps(1, 0, 1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(-1, 1, 1, 1),
  _mm_set_ps(0, 1, 1, 1),
  _mm_set_ps(1, 1, 1, 1),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0),
  _mm_set_ps(0.0, 0.0, 0.0, 0.0)
};


inline void add_increment_old(const uint8 w, const float* input_data, float* out0, float* out1, float* out2, float* out3) {
// inline void add_increment_old(const uint8 w, const float* input_data, float* out) {
  float w0 = static_cast<float>((w >> 6) & 3) - 1.0;
  float w1 = static_cast<float>((w >> 4) & 3) - 1.0;
  float w2 = static_cast<float>((w >> 2) & 3) - 1.0;
  float w3 = static_cast<float>( w       & 3) - 1.0;

  float y0 = w0 * input_data[0];
  float y1 = w1 * input_data[1];
  float y2 = w2 * input_data[2];
  float y3 = w3 * input_data[3];

  // *out += y0 + y1 + y2 +y3;
  *out0 += y0;
  *out1 += y1;
  *out2 += y2;
  *out3 += y3;
}


inline void add_increment(const uint8 w, const float* input_data, float* out0, float* out1, float* out2, float* out3) {
  const uint8 s0 = w >> 6;
  const uint8 s1 = w >> 4;
  const uint8 s2 = w >> 2;
  const uint8 s3 = w;

  const uint8 m0 = s0 & 3;
  const uint8 m1 = s1 & 3;
  const uint8 m2 = s2 & 3;
  const uint8 m3 = s3 & 3;

  const float w0 = static_cast<float>(m0) - 1.0;
  const float w1 = static_cast<float>(m1) - 1.0;
  const float w2 = static_cast<float>(m2) - 1.0;
  const float w3 = static_cast<float>(m3) - 1.0;

  const float y0 = w0 * input_data[0];
  const float y1 = w1 * input_data[1];
  const float y2 = w2 * input_data[2];
  const float y3 = w3 * input_data[3];

  *out0 += y0;
  *out1 += y1;
  *out2 += y2;
  *out3 += y3;
}


// inline void scale_bias_clamp(float* out, const float scale, const float bias, const float clamp) {
//   float val = *out;
//   val *= scale;
//   val += bias;
//   val = std::clamp(val, -clamp, clamp);
//   *out = val;
// }


void tiled_implementation3(const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data
                              , const float* scale, const float* bias, const float clamp
                              ) {
  const long int remainder = chan_in % 4;
  const long int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;

  // tiling loop
  for (int kk = 0 ; kk < chan_out ; kk += BLOCK_SIZE) {

    for (int i = 0 ; i < batch_size ; i++) {

      for (long int j = kk; j < chan_out && j < kk + BLOCK_SIZE; j++) {
        const int out_idx = i*chan_out+j;
        float out0 = 0.0; float out1 = 0.0; float out2 = 0.0; float out3 = 0.0;
        // float out = 0.0;
        for (int k = 0; k < compressed_chan_size_target; k++) {
          const uint8 w = weight_data[j*compressed_chan_size + k];
          const int in_idx_base = i*chan_in + k*4;
          // switch_increment(w, &input_data[in_idx_base], &out);
          add_increment(w, &input_data[in_idx_base], &out0, &out1, &out2, &out3);
          // add_increment_simd(w, &input_data[in_idx_base], &out0, &out1, &out2, &out3);
          // add_increment_old(w, &input_data[in_idx_base], &out);
        }

        const uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
        const int in_idx_base = i*chan_in + chan_in - 4;
        if (remainder > 0) {
          accum_ternary_macro(out0, input_data[in_idx_base], w >> 6);
        }
        if (remainder > 1) {
          accum_ternary_macro(out1, input_data[in_idx_base+1], w >> 4);
        }
        if (remainder > 2) {
          accum_ternary_macro(out2, input_data[in_idx_base+2], w >> 2);
        }

        float out = out0 + out1 + out2 + out3;

        // if (out_idx == 0) {
        //   printf("scale: %f    bias: %f\n", scale[j], bias[j]);
        // }

        // scale_bias_clamp(&out, scale[j], bias[j], clamp);

        out *= scale[j];
        out += bias[j];
        out = std::clamp(out, -clamp, clamp);

        output_data[out_idx] = out;
        // output_data[out_idx] = out0 + out1 + out2 + out3;


      } // output loop

    } // batch loop

  } // tile loop

  // printf("total: %d\n", total);
}


const __m128i shifts = _mm_setr_epi32(6, 4, 2, 0);
const __m128i mask = _mm_set_epi32(3, 3, 3, 3);
const __m128i zeros = _mm_set_epi32(0, 0, 0, 0);
const __m128i ones = _mm_set_epi32(1, 1, 1, 1);
const __m128i twos = _mm_set_epi32(2, 2, 2, 2);


inline void add_increment_simd_old(const uint8 w, const float* input_data, __m128* out) {
  const __m128 input = _mm_loadu_ps(input_data); // input_data[0] is placed in the right (least significant bits) 
  // const __m128i ww = _mm_setr_epi32(w >> 6, w >> 4, w >> 2, w); // beware of loading order, we reverse the order to match the input 
  // __m128i ww = _mm_set_epi32(w, w, w, w);
  __m128i ww = _mm_set1_epi32(w);
  ww = _mm_srav_epi32(ww, shifts);
  const __m128i masked = _mm_and_si128(ww, mask);
  const __m128i centered = _mm_sub_epi32(masked, ones);
  const __m128 converted = _mm_cvtepi32_ps(centered);
  
  // const __m128 res = _mm_mul_ps(input, converted);
  // *out = _mm_add_ps(res, *out);

  *out = _mm_fmadd_ps(input, converted, *out);
}

inline void add_increment_simd(const uint8 w, const float* input_data, __m128* out) {
  const __m128 input = _mm_loadu_ps(input_data); // input_data[0] is placed in the right (least significant bits) 
  __m128 ww = unpack_table[w];
  *out = _mm_fmadd_ps(input, ww, *out);
}


inline void add_increment_simd_maskload(const uint8 w, const float* input_data, __m128* out) {
  // const __m128i ww = _mm_setr_epi32(w >> 6, w >> 4, w >> 2, w); // beware of loading order, we reverse the order to match the input 
  __m128i ww = _mm_set_epi32(w, w, w, w);
  // __m128i ww = _mm_set1_epi32(w);
  ww = _mm_srav_epi32(ww, shifts);
  const __m128i masked = _mm_and_si128(ww, mask);

  const __m128i is_pos = _mm_cmpeq_epi32(masked, twos);
  __m128i is_neg = _mm_cmpeq_epi32(masked, zeros);
  
  const __m128 input_pos = _mm_maskload_ps(input_data, is_pos); // input_data[0] is placed in the right (least significant bits) 
  const __m128 input_neg = _mm_maskload_ps(input_data, is_neg);

  *out = _mm_add_ps(*out, input_pos);
  *out = _mm_sub_ps(*out, input_neg);
}


void tiled_implementation_simd(const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data
                              , const float* scale, const float* bias, const float clamp
                              ) {
  const long int remainder = chan_in % 4;
  const long int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;

  // tiling loop
  for (int kk = 0 ; kk < chan_out ; kk += BLOCK_SIZE) {

    for (int i = 0 ; i < batch_size ; i++) {

      for (long int j = kk; j < chan_out && j < kk + BLOCK_SIZE; j++) {
        const int out_idx = i*chan_out+j;
        
        __m128 out = _mm_setzero_ps();

        // float out = 0.0;
        for (int k = 0; k < compressed_chan_size_target; k++) {
          const uint8 w = weight_data[j*compressed_chan_size + k];
          const int in_idx_base = i*chan_in + k*4;
          // switch_increment(w, &input_data[in_idx_base], &out);
          // add_increment(w, &input_data[in_idx_base], &out0, &out1, &out2, &out3);
          add_increment_simd(w, &input_data[in_idx_base], &out);
          // add_increment_old(w, &input_data[in_idx_base], &out);
        }

        float out_scalar =  _mm_cvtss_f32(_mm_shuffle_ps(out, out, 0));  // Extract the first float from res
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 1));
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 2));
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 3));

        const uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
        const int in_idx_base = i*chan_in + chan_in - 4;
        if (remainder > 0) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base], w >> 6);
        }
        if (remainder > 1) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base+1], w >> 4);
        }
        if (remainder > 2) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base+2], w >> 2);
        }

        out_scalar *= scale[j];
        out_scalar += bias[j];
        out_scalar = std::clamp(out_scalar, -clamp, clamp);

        output_data[out_idx] = out_scalar;
        // output_data[out_idx] = out0 + out1 + out2 + out3;


      } // output loop

    } // batch loop

  } // tile loop

  // printf("total: %d\n", total);
}


using std::thread;


struct thread_data {
  const int batch_size;
  const int chan_in;
  const int chan_out;
  const int compressed_chan_size;
  const float* input_data;
  float* output_data;
  const uint8* weight_data;
  const float* scale;
  const float* bias;
  const float clamp;
};


void thread_matmul(const int idx_start, const int idx_stop, thread_data data) {
  const int chan_in = data.chan_in;
  const int compressed_chan_size = data.compressed_chan_size;
  const int chan_out = data.chan_out;
  const int batch_size = data.batch_size;

  const float* input_data = data.input_data;
  const uint8* weight_data = data.weight_data;
  const float* scale = data.scale;
  const float* bias = data.bias;
  const float clamp = data.clamp;

  float* output_data = data.output_data;


  const long int remainder = chan_in % 4;
  const long int compressed_chan_size_target = remainder == 0? compressed_chan_size : compressed_chan_size - 1;

  // tiling loop
  for (int kk = idx_start ; kk < idx_stop ; kk += BLOCK_SIZE) {

    for (int i = 0 ; i < batch_size ; i++) {

      for (long int j = kk; j < idx_stop && j < kk + BLOCK_SIZE; j++) {
        const int out_idx = i*chan_out+j;
        
        __m128 out = _mm_setzero_ps();

        // float out = 0.0;
        // for (int k = 0; k < compressed_chan_size_target; k++) {
        //   const uint8 w = weight_data[j*compressed_chan_size + k];
        //   const int in_idx_base = i*chan_in + k*4;
        //   // switch_increment(w, &input_data[in_idx_base], &out);
        //   // add_increment(w, &input_data[in_idx_base], &out0, &out1, &out2, &out3);
        //   add_increment_simd(w, &input_data[in_idx_base], &out);
        //   // add_increment_old(w, &input_data[in_idx_base], &out);
        // }

        int w_index = j*compressed_chan_size;
        int in_idx_base = i*chan_in;
        for (int k = 0; k < compressed_chan_size_target; k++, w_index++, in_idx_base += 4) {
          add_increment_simd(weight_data[w_index], &input_data[in_idx_base], &out);
        }

        float out_scalar =  _mm_cvtss_f32(_mm_shuffle_ps(out, out, 0));  // Extract the first float from res
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 1));
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 2));
        out_scalar += _mm_cvtss_f32(_mm_shuffle_ps(out, out, 3));

        const uint8 w = weight_data[j*compressed_chan_size + compressed_chan_size - 1];
        const int in_idx_base = i*chan_in + chan_in - 4;
        if (remainder > 0) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base], w >> 6);
        }
        if (remainder > 1) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base+1], w >> 4);
        }
        if (remainder > 2) {
          accum_ternary_macro(out_scalar, input_data[in_idx_base+2], w >> 2);
        }

        out_scalar *= scale[j];
        out_scalar += bias[j];
        out_scalar = std::clamp(out_scalar, -clamp, clamp);

        output_data[out_idx] = out_scalar;
        // output_data[out_idx] = out0 + out1 + out2 + out3;


      } // output loop

    } // batch loop

  } // tile loop
}


void tiled_implementation_threaded(const int reco_num_threads, const int batch_size, const int chan_in, const int chan_out, const int compressed_chan_size,
                              const float* input_data, float* output_data, const uint8* weight_data,
                              const float* scale, const float* bias, const float clamp
                              ) {
  
  int n_threads = 1;
  if (reco_num_threads <= 0) {
    n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
  }
  else {
    n_threads = reco_num_threads;
  }
  
  const int thread_block_size = chan_out / n_threads;

  thread threads[n_threads];

  for (int i = 0; i < n_threads - 1; i++) {
    const int start_idx = i * thread_block_size;
    const int end_idx = (i+1) * thread_block_size;
    thread_data data = {batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data, scale, bias, clamp};
    threads[i] = thread(thread_matmul, start_idx, end_idx, data);
  }

  const int start_idx = (n_threads - 1) * thread_block_size;
  const int end_idx = chan_out;
  thread_data data = {batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data, scale, bias, clamp};
  threads[n_threads - 1] = thread(thread_matmul, start_idx, end_idx, data);

  for (thread &t : threads) {
    t.join();
  }

  // printf("total: %d\n", total);
}


TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* input = GetInput(context, node, 0);
  auto* weights = GetInput(context, node, 1);
  auto* scale = GetInput(context, node, 2);
  auto* bias = GetInput(context, node, 3);
  auto* clamp = GetInput(context, node, 4);
  auto* output = GetOutput(context, node, 0);

  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape weights_shape = GetTensorShape(weights);
  RuntimeShape output_shape = GetTensorShape(output);

  const float* input_data = GetTensorData<float>(input);
  const uint8* weight_data = GetTensorData<uint8>(weights);
  const float* scale_data = GetTensorData<float>(scale);
  const float* bias_data = GetTensorData<float>(bias);
  const float clamp_data = abs(GetTensorData<float>(clamp)[0]);
  float* output_data = GetTensorData<float>(output);
  
  // printf("first weight: %i\n", weight_data[0]);

  int compressed_chan_size = weights_shape.Dims(1);

  int chan_in = input_shape.Dims(input_shape.DimensionsCount()-1);
  int chan_out = output_shape.Dims(output_shape.DimensionsCount()-1);
  int batch_size = 1;
  for (int i = 0; i < output_shape.DimensionsCount(); i++) {
    int dim = output_shape.Dims(i);
    // printf("Dimension %i: %i\n", i, dim);
    if (i < output_shape.DimensionsCount() - 1) {
      batch_size = batch_size * dim;
    }
  }

  // printf("batch size: %d, chan in: %d, chan out: %d\n", batch_size, chan_in, chan_out);

  // printf("clamp: %f, scale shape: %d\n", clamp_data, scale->dims->data[0]);

  // reference_implementation(batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data);
  // tiled_implementation3(batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data, scale_data, bias_data, clamp_data);
  // tiled_implementation_simd(batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data, scale_data, bias_data, clamp_data);
  tiled_implementation_threaded(context->recommended_num_threads, batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data, scale_data, bias_data, clamp_data);
  // tiled_implementation3(batch_size, chan_in, chan_out, compressed_chan_size, input_data, output_data, weight_data);

  return kTfLiteOk;
}

} // namespace ternary

TfLiteRegistration* Register_TERNARY_OPT() {
  // printf("registering Ternary Matmul\n");
  static TfLiteRegistration r = {
      ternary::Init,
      ternary::Free,
      ternary::Prepare,
      ternary::Eval};
  return &r;
}


}  // namespace tflite
}  // namespace compute_engine

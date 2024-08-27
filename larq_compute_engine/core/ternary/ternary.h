#ifndef LARQ_COMPUTE_ENGINE_CORE_TERNARY
#define LARQ_COMPUTE_ENGINE_CORE_TERNARY

#include "tensorflow/core/framework/op_kernel.h"
#include <thread>

using namespace tensorflow;

// using std::thread;



void unpack_ternary_chunk(const int start_idx, const int end_idx, const uint8* packed, float* out, const int unpacked_chan_in, const int chan_out) {
  int packed_chan_in = unpacked_chan_in / 4;
  const int remainder = unpacked_chan_in % 4;

  if (remainder != 0) {
    ++packed_chan_in;
  }

  const int target_packed_size = remainder == 0 ? packed_chan_in : packed_chan_in - 1;

  // int packed_tot_size = packed_chan_in * chan_out;
  // int tot_size = unpacked_chan_in * chan_out;

  // printf("Will iterate for %ix%i\n", chan_out, target_packed_size);

  for (int i = start_idx; i < end_idx ; i++) {
    for (int j = 0 ; j < target_packed_size; j++) {
      const int idx_in = i * packed_chan_in + j;
      const int idx_out_base = i * unpacked_chan_in + (j * 4);
      const uint8 pw = packed[idx_in];
      // printf("packed: %i, will write %f to idx %i\n", packed[idx_in], ((float) ((packed[idx_in]) & 3)) - 1.0, idx_out_base);
      out[idx_out_base  ] = static_cast<float>((pw >> 6) & 3) - 1.0;
      out[idx_out_base+1] = static_cast<float>((pw >> 4) & 3) - 1.0;
      out[idx_out_base+2] = static_cast<float>((pw >> 2) & 3) - 1.0;
      out[idx_out_base+3] = static_cast<float>((pw     ) & 3) - 1.0;
    }

    const int idx_in = i * packed_chan_in - 1;
    const int idx_out_base = i * unpacked_chan_in + unpacked_chan_in - remainder;
    const uint pw = packed[idx_in];
    if (remainder > 0) out[idx_out_base  ] = static_cast<float>((pw >> 6) & 3) - 1.0;
    if (remainder > 1) out[idx_out_base+1] = static_cast<float>((pw >> 4) & 3) - 1.0;
    if (remainder > 2) out[idx_out_base+2] = static_cast<float>((pw >> 2) & 3) - 1.0;

    
  }
}


void unpack_ternary_threaded(const int num_threads, const uint8* packed, float* out, const int unpacked_chan_in, const int chan_out) {
  int packed_chan_in = unpacked_chan_in / 4;
  const int remainder = unpacked_chan_in % 4;

  if (remainder != 0) {
    ++packed_chan_in;
  }

  const int target_packed_size = remainder == 0 ? packed_chan_in : packed_chan_in - 1;

  int n_threads = 1;
  if (num_threads <= 0) {
    n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
  }
  else {
    n_threads = num_threads;
  }

  int thread_block_size = chan_out / n_threads;

  std::thread threads[n_threads];

  for (int i = 0; i < n_threads - 1; i++) {
    const int start_idx = i * thread_block_size;
    const int end_idx = (i+1) * thread_block_size;
    threads[i] = std::thread(unpack_ternary_chunk, start_idx, end_idx, packed, out, unpacked_chan_in, chan_out);
  }

  const int start_idx = (n_threads - 1) * thread_block_size;
  const int end_idx = chan_out;
  threads[n_threads - 1] = std::thread(unpack_ternary_chunk, start_idx, end_idx, packed, out, unpacked_chan_in, chan_out);

  for (std::thread &t : threads) {
    t.join();
  }
}


void unpack_ternary(const uint8* packed, float* out, const int unpacked_chan_in, const int chan_out) {
  int packed_chan_in = unpacked_chan_in / 4;
  const int remainder = unpacked_chan_in % 4;

  if (remainder != 0) {
    ++packed_chan_in;
  }

  const int target_packed_size = remainder == 0 ? packed_chan_in : packed_chan_in - 1;

  // int packed_tot_size = packed_chan_in * chan_out;
  // int tot_size = unpacked_chan_in * chan_out;

  // printf("Will iterate for %ix%i\n", chan_out, target_packed_size);

  for (int i = 0; i < chan_out ; i++) {
    for (int j = 0 ; j < target_packed_size; j++) {
      const int idx_in = i * packed_chan_in + j;
      const int idx_out_base = i * unpacked_chan_in + (j * 4);
      const uint8 pw = packed[idx_in];
      // printf("packed: %i, will write %f to idx %i\n", packed[idx_in], ((float) ((packed[idx_in]) & 3)) - 1.0, idx_out_base);
      out[idx_out_base  ] = static_cast<float>((pw >> 6) & 3) - 1.0;
      out[idx_out_base+1] = static_cast<float>((pw >> 4) & 3) - 1.0;
      out[idx_out_base+2] = static_cast<float>((pw >> 2) & 3) - 1.0;
      out[idx_out_base+3] = static_cast<float>((pw     ) & 3) - 1.0;
    }

    const int idx_in = i * packed_chan_in - 1;
    const int idx_out_base = i * unpacked_chan_in + unpacked_chan_in - remainder;
    const uint pw = packed[idx_in];
    if (remainder > 0) out[idx_out_base  ] = static_cast<float>((pw >> 6) & 3) - 1.0;
    if (remainder > 1) out[idx_out_base+1] = static_cast<float>((pw >> 4) & 3) - 1.0;
    if (remainder > 2) out[idx_out_base+2] = static_cast<float>((pw >> 2) & 3) - 1.0;

    
  }
}

#endif
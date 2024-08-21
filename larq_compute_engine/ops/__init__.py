import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

from tensorflow import keras

custom_ops_mod = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_ops.so'))

custom_ops_wrapper = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_ops_tmp.so'))

ternary_matmul = custom_ops_mod.ternary_matmul

unpack_ternary = custom_ops_mod.unpack_ternary


class Ternary(keras.layers.Layer):
    def call(self, x, w):
        return ternary_matmul(x, w)
    


def pack_fn(w):
    size = tf.shape(w)[-1]
    remainder = size % 4
    to_add = (4 - remainder) % 4
    
    if to_add != 0:
        # print("adding padding, previous size:", size)
        w = tf.concat([w, tf.zeros((w.shape[0],to_add), dtype=w.dtype)], axis=-1)

    size = tf.shape(w)[-1]
    slice_size = size // 4

    # print("slice size:", slice_size)
    
    # packed = tf.zeros((*w.shape[:-1], slice_size), dtype=tf.uint8)

    tensors = []

    for j in range(slice_size):
        s = w[..., j*4:(j+1)*4]
        val_idx = s + 1
        packed = tf.zeros(w.shape[:-1], dtype=tf.uint8)
        for i in range(4):
            val = tf.bitwise.left_shift(tf.cast(val_idx[:, (3-i)], dtype=tf.uint8), i*2)
            packed = tf.bitwise.bitwise_or(packed, val)
        tensors.append(packed)

    for i in range(remainder):
        packed = tensors[-1]
        packed = tf.bitwise.bitwise_or(packed, tf.bitwise.left_shift(tf.constant(1, dtype=tf.uint8), i*2))
        tensors[-1] = packed

    packed = tf.stack(tensors, -1)
    return packed


__all__ = [ternary_matmul, Ternary, pack_fn, unpack_ternary]

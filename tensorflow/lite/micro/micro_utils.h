/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// Returns number of elements in the shape array.

int ElementCount(const TfLiteIntArray& dims);

// size_t EvalTensorBytes(const TfLiteEvalTensor* tensor);

// C++11 does not support constexpr max; hence, use ternary conditional to
// create our own constexpr Max function.
constexpr int Max(int a, int b) { return a >= b ? a : b; }

// Converts a float value into a quantized value.  Note that large values (close
// to max int and min int) may see significant error due to a lack of floating
// point granularity for large values.
template <typename T>
T FloatToQuantizedType(const float value, const float scale, int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  result =
      std::max(static_cast<int32_t>(std::numeric_limits<T>::min()), result);
  result =
      std::min(static_cast<int32_t>(std::numeric_limits<T>::max()), result);
  return result;
}

template <typename T>
T FloatToSymmetricQuantizedType(const float value, const float scale) {
  // 64-bit values are required since 8x16 conv accumulates to int64, meaning
  // an int64 bias is required.
  std::int64_t result = round(value / scale);
  result = std::max(
      static_cast<std::int64_t>(std::numeric_limits<T>::min() + 1), result);
  result = std::min(static_cast<std::int64_t>(std::numeric_limits<T>::max()),
                    result);
  return result;
}

// Helper methods to quantize arrays of floats to the desired format.
//
// There are several key flavors of quantization in TfLite:
//        asymmetric symmetric  per channel
// int8_t  |     X    |    X    |     X      |
// uint8_t |     X    |    X    |            |
// int16_t |     X    |         |            |
// int32_t |          |    X    |     X      |
//
// The per-op quantization spec can be found here:
// https://www.tensorflow.org/lite/performance/quantization_spec
template <typename T>
void Quantize(const float* input, T* output, int num_elements, float scale,
              int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToQuantizedType<T>(input[i], scale, zero_point);
  }
}

template <typename T>
void SymmetricQuantize(const float* input, T* output, int num_elements,
                       float scale) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToSymmetricQuantizedType<T>(input[i], scale);
  }
}

template <typename T>
void Dequantize(const T* values, const int size, const float scale,
                int zero_point, float* dequantized_values) {
  for (int i = 0; i < size; ++i) {
    dequantized_values[i] = (values[i] - zero_point) * scale;
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

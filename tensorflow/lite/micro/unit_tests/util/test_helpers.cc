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

#include "util/test_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace testing {

inline int QMinFromTfLiteType(TfLiteType type) {
  if (type == kTfLiteInt4) {
    return -8;
  } else {
    return std::numeric_limits<int8_t>::min();
  }
}

inline int QMaxFromTfLiteType(TfLiteType type) {
  if (type == kTfLiteInt4) {
    return 7;
  } else {
    return std::numeric_limits<int8_t>::max();
  }
}

void SignedSymmetricPerChannelQuantize(
    const float* values, TfLiteIntArray* dims, int quantized_dimension,
    int8_t* quantized_values, float* scaling_factors, TfLiteType type) {
  int input_size = ElementCount(*dims);
  int channel_count = dims->data[quantized_dimension];
  int per_channel_size = input_size / channel_count;

  int stride;
  int channel_stride;

  int qmin = QMinFromTfLiteType(type);
  int qmax = QMaxFromTfLiteType(type);

  if (quantized_dimension == 0) {
    stride = 1;
    channel_stride = per_channel_size;
  } else if (quantized_dimension == 3) {
    stride = channel_count;
    channel_stride = 1;
  } else {
    MicroPrintf("quantized dimension must be 0 or 3");
    std::abort();
  }

  // Calculate scales for each channel.
  for (int channel = 0; channel < channel_count; channel++) {
    float min = 0;
    float max = 0;

    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      min = fminf(min, values[idx]);
      max = fmaxf(max, values[idx]);
    }
    scaling_factors[channel] = fmaxf(fabs(min), fabs(max)) / qmax;
    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      const int32_t quantized_value =
          static_cast<int32_t>(roundf(values[idx] / scaling_factors[channel]));
      // Clamp: just in case some odd numeric offset.
      quantized_values[idx] = fminf(qmax, fmaxf(qmin + 1, quantized_value));
    }
  }
}

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable, TfLiteType tensor_weight_type) {
  int channel_count = dims->data[quantized_dimension];

  scales[0] = static_cast<float>(channel_count);
  zero_points[0] = channel_count;

  SignedSymmetricPerChannelQuantize(input, dims, quantized_dimension, quantized,
                                    &scales[1], tensor_weight_type);

  for (int i = 0; i < channel_count; i++) {
    zero_points[i + 1] = 0;
  }

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;
  TfLiteTensor result =
      CreateTensor(quantized, dims, is_variable, tensor_weight_type);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  return result;
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int16_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale, bool is_variable) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);

  // Quantized int16_t tensors always have a zero point of 0, since the range of
  // int16_t values is large, and because zero point costs extra cycles during
  // processing.
  TfLiteTensor result =
      CreateQuantizedTensor(quantized, dims, bias_scale, 0, is_variable);
  return result;
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int32_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale, bool is_variable) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);

  // Quantized int32_t tensors always have a zero point of 0, since the range of
  // int32_t values is large, and because zero point costs extra cycles during
  // processing.
  TfLiteTensor result =
      CreateQuantizedTensor(quantized, dims, bias_scale, 0, is_variable);
  return result;
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data,
                                       std::int64_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale, bool is_variable) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);

  // Quantized int32_t tensors always have a zero point of 0, since the range of
  // int32_t values is large, and because zero point costs extra cycles during
  // processing.
  TfLiteTensor result =
      CreateQuantizedTensor(quantized, dims, bias_scale, 0, is_variable);
  return result;
}

TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, int32_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable) {
  return CreatePerChannelQuantizedBiasTensor<int32_t>(
      input, quantized, dims, input_scale, weight_scales, scales, zero_points,
      affine_quant, quantized_dimension, is_variable);
}

TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, std::int64_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable) {
  return CreatePerChannelQuantizedBiasTensor<std::int64_t>(
      input, quantized, dims, input_scale, weight_scales, scales, zero_points,
      affine_quant, quantized_dimension, is_variable);
}


}  // namespace testing
}  // namespace tflite

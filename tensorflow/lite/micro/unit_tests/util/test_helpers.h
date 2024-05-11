/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_TEST_HELPER_H_
#define TENSORFLOW_LITE_MICRO_TEST_HELPER_H_

// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace testing {

inline TfLiteIntArray* IntArrayFromInts(const int* int_array) {
  return reinterpret_cast<TfLiteIntArray*>(const_cast<int*>(int_array));
}

// Create a TfLiteFloatArray from an array of floats.  The first element in the
// supplied array must be the size of the array expressed as a float.
inline TfLiteFloatArray* FloatArrayFromFloats(const float* floats) {
  static_assert(sizeof(float) == sizeof(int), "assumes sizeof(float) == sizeof(int) to perform casting");
  int size = static_cast<int>(floats[0]);
  *reinterpret_cast<int32_t*>(const_cast<float*>(floats)) = size;
  return reinterpret_cast<TfLiteFloatArray*>(const_cast<float*>(floats));
}

template <typename T>
TfLiteTensor CreateTensor(const T* data, TfLiteIntArray* dims,
                          const bool is_variable = false,
                          TfLiteType type = kTfLiteNoType) {
  TfLiteTensor result;
  result.dims = dims;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.data.data = const_cast<T*>(data);
  result.bytes = ElementCount(*dims) * sizeof(T);
  result.data.data = const_cast<T*>(data);
  result.type = typeToTfLiteType<T>();
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const T* data, TfLiteIntArray* dims,
                                   const float scale, const int zero_point = 0,
                                   const bool is_variable = false,
                                   TfLiteType type = kTfLiteNoType) {
  TfLiteTensor result = CreateTensor(data, dims, is_variable, type);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const float* input, T* quantized,
                                   TfLiteIntArray* dims, float scale,
                                   int zero_point, bool is_variable = false,
                                   TfLiteType type = kTfLiteNoType) {
  int input_size = ElementCount(*dims);
  tflite::Quantize(input, quantized, input_size, scale, zero_point);
  return CreateQuantizedTensor(quantized, dims, scale, zero_point, is_variable,
                               type);
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int16_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale,
                                       bool is_variable = false);

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int32_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale,
                                       bool is_variable = false);

TfLiteTensor CreateQuantizedBiasTensor(const float* data,
                                       std::int64_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale,
                                       bool is_variable = false);
                                       
template <typename T>
void SymmetricPerChannelQuantize(const float* input, T* output,
                                 int num_elements, int num_channels,
                                 float* scales) {
  int elements_per_channel = num_elements / num_channels;
  for (int i = 0; i < num_channels; i++) {
    for (int j = 0; j < elements_per_channel; j++) {
      output[i * elements_per_channel + j] = FloatToSymmetricQuantizedType<T>(
          input[i * elements_per_channel + j], scales[i]);
    }
  }
}

// Quantizes int32_t bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
template <typename T>
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, T* quantized, TfLiteIntArray* dims, float input_scale,
    float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable) {
  int input_size = ElementCount(*dims);
  int num_channels = dims->data[quantized_dimension];
  // First element is reserved for array length
  zero_points[0] = num_channels;
  scales[0] = static_cast<float>(num_channels);
  float* scales_array = &scales[1];
  for (int i = 0; i < num_channels; i++) {
    scales_array[i] = input_scale * weight_scales[i];
    zero_points[i + 1] = 0;
  }

  SymmetricPerChannelQuantize<T>(input, quantized, input_size, num_channels,
                                 scales_array);

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(quantized, dims, is_variable);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  return result;
}

// Quantizes int32_t bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, int32_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable = false);

// Quantizes int64_t bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, std::int64_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable = false);

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable = false,
    TfLiteType tensor_weight_type = kTfLiteNoType);

// Derives the asymmetric quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) /
         static_cast<float>((std::numeric_limits<T>::max() * 1.0) -
                            std::numeric_limits<T>::min());
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(roundf(-min / ScaleFromMinMax<T>(min, max)));
}



}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_FAKE_MICRO_CONTEXT_H_

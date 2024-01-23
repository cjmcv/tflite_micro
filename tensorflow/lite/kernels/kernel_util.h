/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

#include <stdint.h>

#include <limits>
#ifndef TF_LITE_STATIC_MEMORY
#include <string>
#endif  // TF_LITE_STATIC_MEMORY

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#ifndef NDEBUG
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#endif

namespace tflite {

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
  return t->dims->data[dim];
}

inline int NumInputs(const TfLiteNode* node) {
  return node->inputs == nullptr ? 0 : node->inputs->size;
}
inline int NumOutputs(const TfLiteNode* node) {
  return node->outputs == nullptr ? 0 : node->outputs->size;
}

inline int64_t NumElements(const int* dims, int num_dims) {
  int64_t count = 1;
  for (int i = 0; i < num_dims; ++i) {
#ifndef NDEBUG
    if (count <= 0) {
      break;
    }
    // Check that number of elements can fit in 32 bit int. Most of tflite
    // assumes the result of `NumElements` is < MAX_INT and static or implicit
    // casts to `int32_t` without any checks. It is more meaningful to check
    // that the result fits into 32 bits than for standard overflow on 64 bit
    // type.
    TFLITE_DCHECK(dims[i] < std::numeric_limits<int>::max() / count);
#endif
    count *= dims[i];
  }
  return count;
}

inline int64_t NumElements(const TfLiteIntArray* dims) {
  return NumElements(dims->data, dims->size);
}

inline int64_t NumElements(const TfLiteTensor* t) {
  return NumElements(t->dims);
}

// Determines whether tensor is constant.
// TODO(b/138199592): Introduce new query which checks for constant OR
// persistent-read-only, which would be useful for most tensor kernels that
// are potentially dynamic based on the input tensor value availability at the
// time of prepare.
inline bool IsConstantTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteMmapRo;
}

inline bool IsConstantOrPersistentTensor(const TfLiteTensor* tensor) {
  return IsConstantTensor(tensor) ||
         (tensor->allocation_type == kTfLitePersistentRo);
}

// Determines whether tensor is dynamic. Note that a tensor can be non-const and
// not dynamic. This function specifically checks for a dynamic tensor.
inline bool IsDynamicTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteDynamic;
}

// Check dimensionality match and populate OpData for Conv and DepthwiseConv.
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift);

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels);

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier);

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation tensor.
TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max);

// Calculates the useful range of an activation layer given its activation
// tensor.a
template <typename T>
void CalculateActivationRange(TfLiteFusedActivation activation,
                              T* activation_min, T* activation_max) {
  if (activation == kTfLiteActRelu) {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0;
    *activation_max = 6;
  } else if (activation == kTfLiteActReluN1To1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2);

// Return the size of given type in bytes. Return 0 in case of string.
int TfLiteTypeGetSize(TfLiteType type);

// // Whether the current platform is mobile (Android or iOS).
// bool IsMobilePlatform();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

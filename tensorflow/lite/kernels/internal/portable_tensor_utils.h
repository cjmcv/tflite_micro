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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Multiplies a matrix with a scalar and reduce the result on each row to a
// scalar.
// Parameters:
//     - matrix: matrix of size n_row * n_col
//     - scalar: the scalar that is multiplied to each element in the matrix
//     - n_row:  the row count of the matrix
//     - n_col:  the column count of the matrix
//     - output: the 32bit output
// Note: We do not need saturation because the int8 * int8 is safe from overflow
// in (2^31-1) / (2^14) = 131072, which is bigger than the n_row. Non-zero
// initial output value is not exceptionally large.
void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                    int32_t n_row, int32_t n_col,
                                    int32_t* output);
                                    
// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size);

// Same as above but input/output is 32 bit integer.
void ReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size);

// Same as above but input is 8 bit integer.
void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size);

// Apply Rectified Linear to elements of a vector.
void ApplyReluToVector(const float* __restrict__ vector, int v_size,
                       float* __restrict__ result);

// Apply Rectified Linear 1 (cap to [-1;1]) to elements of a vector
void ApplyRelu1ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result);

// Apply Rectified Linear 6 (cap to [0;6]) to elements of a vector
void ApplyRelu6ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result);

// Apply signbit to elements of a vector
void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                          float* __restrict__ result);

// Unpack or inflate `src_buffer` by taking each element and splitting it as
// two elements into `dst_buffer`.
// Parameters:
//   src_buffer   : Densely packed buffer containing int4 values
//   num_elements : Number of elements stored in the buffer. Note that this can
//                  be smaller than the size of `src_buffer` by 1 if it's odd,
//                  in which case the last nibble in `src_buffer` is ignored.
//                  This should be equal to the size of `dst_buffer`.
//   dst_buffer   : Buffer to unpack into. Should be allocated by the caller.
//                  Size should be at least `num_elements`.
// Notes:
//   For example, given `src_buffer = {0x12, 0x34};`, calling this function
//   will return `dst_buffer = {0x02, 0x01, 0x04, 0x03}`.
void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer);

}  // namespace tensor_utils

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_

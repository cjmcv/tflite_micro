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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_

#include <algorithm>
#include <cstdint>

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation.
class CpuBackendContext;

namespace tensor_utils {

void PortableMatrixScalarMultiplyAccumulate(const int8_t* matrix,
                                            int32_t scalar, int32_t n_row,
                                            int32_t n_col, int32_t* output);

// Reduce-sum on a vector:
// input_vector: pointer to input vector.
// output_vector: pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
template <typename INPUT, typename OUTPUT>
void PortableReductionSumVector(const INPUT* input_vector,
                                OUTPUT* output_vector, int output_size,
                                int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    OUTPUT result = 0;
    for (int r = 0; r < reduction_size; r++) {
      result += input_vector[r];
    }
    output_vector[o] = result;
    input_vector += reduction_size;
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_

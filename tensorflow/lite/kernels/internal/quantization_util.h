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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// // Given the min and max values of a float array, return
// // reasonable quantization parameters to use for this array.
// template <typename T>
// QuantizationParams ChooseQuantizationParams(double rmin, double rmax,
//                                             bool narrow_range) {
//   const T qmin = std::numeric_limits<T>::min() + (narrow_range ? 1 : 0);
//   const T qmax = std::numeric_limits<T>::max();
//   const double qmin_double = qmin;
//   const double qmax_double = qmax;
//   // 0 should always be a representable value. Let's assume that the initial
//   // min,max range contains 0.
//   TFLITE_DCHECK_LE(rmin, 0.);
//   TFLITE_DCHECK_GE(rmax, 0.);
//   if (rmin == rmax) {
//     // Special case where the min,max range is a point. Should be {0}.
//     TFLITE_DCHECK_EQ(rmin, 0.);
//     TFLITE_DCHECK_EQ(rmax, 0.);
//     QuantizationParams quantization_params;
//     quantization_params.zero_point = 0;
//     quantization_params.scale = 0.;
//     return quantization_params;
//   }

//   // General case.
//   //
//   // First determine the scale.
//   const double scale = (rmax - rmin) / (qmax_double - qmin_double);

//   // Zero-point computation.
//   // First the initial floating-point computation. The zero-point can be
//   // determined from solving an affine equation for any known pair
//   // (real value, corresponding quantized value).
//   // We know two such pairs: (rmin, qmin) and (rmax, qmax).
//   // The arithmetic error on the zero point computed from either pair
//   // will be roughly machine_epsilon * (sum of absolute values of terms)
//   // so we want to use the variant that adds the smaller terms.
//   const double zero_point_from_min = qmin_double - rmin / scale;
//   const double zero_point_from_max = qmax_double - rmax / scale;
//   const double zero_point_from_min_error =
//       std::abs(qmin_double) + std::abs(rmin / scale);
//   const double zero_point_from_max_error =
//       std::abs(qmax_double) + std::abs(rmax / scale);

//   const double zero_point_double =
//       zero_point_from_min_error < zero_point_from_max_error
//           ? zero_point_from_min
//           : zero_point_from_max;

//   // Now we need to nudge the zero point to be an integer
//   // (our zero points are integer, and this is motivated by the requirement
//   // to be able to represent the real value "0" exactly as a quantized value,
//   // which is required in multiple places, for example in Im2col with SAME
//   // padding).
//   T nudged_zero_point = 0;
//   if (zero_point_double < qmin_double) {
//     nudged_zero_point = qmin;
//   } else if (zero_point_double > qmax_double) {
//     nudged_zero_point = qmax;
//   } else {
//     nudged_zero_point = static_cast<T>(round(zero_point_double));
//   }
//   // The zero point should always be in the range of quantized value,
//   // [qmin, qmax].
//   TFLITE_DCHECK_GE(nudged_zero_point, qmin);
//   TFLITE_DCHECK_LE(nudged_zero_point, qmax);

//   // Finally, store the result nudged quantization params.
//   QuantizationParams quantization_params;
//   quantization_params.zero_point = nudged_zero_point;
//   quantization_params.scale = scale;
//   return quantization_params;
// }

// template <typename T>
// QuantizationParams ChooseQuantizationParams(double rmin, double rmax) {
//   return ChooseQuantizationParams<T>(rmin, rmax, false);
// }

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of NEGATIVE its exponent ---
// this is intended as a RIGHT-shift.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);

// Splits a double input value into a returned fraction, and a shift value from
// the exponent, using only bitwise and integer operations to support
// microcontrollers and other environments without floating-point support.
//
// This is designed to be a replacement for how std::frexp() is used within the
// QuantizeMultiplier() function, and so has a different signature than the
// standard version, returning a 64-bit integer rather than a double. This
// result has a maximum value of 1<<31, with the fraction expressed as a
// proportion of that maximum.
//
// std::frexp() returns NaNs and infinities unmodified, but since we're
// returning integers that can't represent those values, instead we return
// a shift of std::numeric_limits<int>::max() for all bad numbers, with an int64
// result of 0 for NaNs, std:numeric_limits<int64_t>::max() for +INFINITY, and
// std::numeric_limits<int64_t>::min() for -INFINITY. Denormalized inputs will
// result in return values that end up truncating some bits at the end,
// reflecting the loss of precision inherent in denormalization.
int64_t IntegerFrExp(double input, int* shift);

// Converts an integer fraction in the format produced by IntegerFrExp (where
// 0x40000000 is 1.0) and an exponent shift (between -1022 and +1022) into an
// IEEE binary64 double format result. The implementation uses only integer and
// bitwise operators, so no floating point hardware support or emulation is
// needed. This is here so quantized operations can run non-time-critical
// preparation calculations on microcontrollers and other platforms without
// float support.
double DoubleFromFractionAndShift(int64_t fraction, int shift);

// Performs a multiplication of two numbers in double format, using only integer
// and bitwise instructions. This is aimed at supporting housekeeping functions
// for quantized operations on microcontrollers without floating-point hardware.
double IntegerDoubleMultiply(double a, double b);

// Returns -1 if a is less than b, 0 if a and b are equal, and +1 if a is
// greater than b. It is implemented using only integer and logical instructions
// so that it can be easily run on microcontrollers for quantized operations.
int IntegerDoubleCompare(double a, double b);

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);
// Like PreprocessSoftmaxScaling, but inverse scaling factors also calculated.
void PreprocessLogSoftmaxScalingExp(double beta, double input_scale,
                                    int input_integer_bits,
                                    int32_t* quantized_multiplier,
                                    int* left_shift,
                                    int32_t* reverse_scaling_divisor,
                                    int* reverse_scaling_left_shift);
// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference in
// Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits = 31);

// If x is approximately a power of two (with any positive or negative
// exponent), stores that exponent (i.e. log2(x)) in *log2_result, otherwise
// returns false.
bool CheckedLog2(const float x, int* log2_result);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

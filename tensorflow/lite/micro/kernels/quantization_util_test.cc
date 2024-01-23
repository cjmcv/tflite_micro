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

#include "tensorflow/lite/kernels/internal/quantization_util.h"

#include "tensorflow/lite/micro/testing/micro_test.h"


TF_LITE_MICRO_TESTS_BEGIN

// // Example taken from http://www.tensorflow.org/performance/quantization
// //
// //  Quantized | Float
// //  --------- | -----
// //  0         | -10.0
// //  255       | 30.0
// //  128       | 10.0
// TF_LITE_MICRO_TEST(QuantizationUtilTest_ChooseQuantizationParams) {
//   tflite::QuantizationParams qp =
//       tflite::ChooseQuantizationParams<uint8_t>(-10.0, 30.0);
//   TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.156863, 1e-5);
//   TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 64);
// }

// TF_LITE_MICRO_TEST(
//     QuantizationUtilTest_ChooseQuantizationParamsZeroPointOnMinBoundary) {
//   tflite::QuantizationParams qp =
//       tflite::ChooseQuantizationParams<uint8_t>(0.0, 30.0);
//   TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.117647, 1e-5);
//   TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 0);
// }

// TF_LITE_MICRO_TEST(
//     QuantizationUtilTest_ChooseQuantizationParamsEmptyRangeZero) {
//   tflite::QuantizationParams qp =
//       tflite::ChooseQuantizationParams<uint8_t>(0.0, 0.0);
//   TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.0, 1e-5);
//   TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 0);
// }

// TF_LITE_MICRO_TEST(
//     QuantizationUtilTest_ChooseQuantizationParamsZeroPointOnMaxBoundary) {
//   tflite::QuantizationParams qp =
//       tflite::ChooseQuantizationParams<uint8_t>(-10.0, 0.0);
//   TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.039216, 1e-5);
//   TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 255);
// }

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerFrExp) {
  int shift;
  int64_t result = tflite::IntegerFrExp(0.0, &shift);
  TF_LITE_MICRO_EXPECT_EQ(0, result);
  TF_LITE_MICRO_EXPECT_EQ(0, shift);

  result = tflite::IntegerFrExp(1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0x40000000, result, 1ll);
  TF_LITE_MICRO_EXPECT_EQ(1, shift);

  result = tflite::IntegerFrExp(0.25, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0x40000000, result, 1ll);
  TF_LITE_MICRO_EXPECT_EQ(-1, shift);

  result = tflite::IntegerFrExp(-1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(-(1 << 30), result, 1ll);
  TF_LITE_MICRO_EXPECT_EQ(1, shift);

  result = tflite::IntegerFrExp(123.45, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(2071147315, result, 1ll);
  TF_LITE_MICRO_EXPECT_EQ(7, shift);

  result = tflite::IntegerFrExp(static_cast<double>(NAN), &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0, result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);

  result = tflite::IntegerFrExp(static_cast<double>(INFINITY), &shift);
  TF_LITE_MICRO_EXPECT_NEAR(std::numeric_limits<int64_t>::max(), result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);

  result = tflite::IntegerFrExp(-static_cast<double>(INFINITY), &shift);
  TF_LITE_MICRO_EXPECT_NEAR(std::numeric_limits<int64_t>::min(), result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerFrExpVersusDouble) {
  int shift;
  int32_t result = tflite::IntegerFrExp(0.0, &shift);
  TF_LITE_MICRO_EXPECT_EQ(result, 0);
  TF_LITE_MICRO_EXPECT_EQ(shift, 0);

  int double_shift;
  double double_result = std::frexp(0.0, &double_shift);
  TF_LITE_MICRO_EXPECT_EQ(double_result, 0);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 0);

  result = tflite::IntegerFrExp(1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, 0x40000000, 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, 1);
  double_result = std::frexp(1.0, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 1);

  result = tflite::IntegerFrExp(0.25, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, 0x40000000, 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, -1);
  double_result = std::frexp(0.25, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, -1);

  result = tflite::IntegerFrExp(-1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, -(1 << 30), 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, 1);
  double_result = std::frexp(-1.0, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, -0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 1);

  result = tflite::IntegerFrExp(123.45, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, (0.964453 * (1LL << 31)), 1000);
  TF_LITE_MICRO_EXPECT_EQ(shift, 7);
  double_result = std::frexp(123.45, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.964453, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 7);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_DoubleFromFractionAndShift) {
  double result = tflite::DoubleFromFractionAndShift(0, 0);
  TF_LITE_MICRO_EXPECT_EQ(0, result);

  result = tflite::DoubleFromFractionAndShift(0x40000000, 1);
  TF_LITE_MICRO_EXPECT_NEAR(1.0, result, 1e-5);

  result = tflite::DoubleFromFractionAndShift(0x40000000, 2);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, result, 1e-5);

  int shift;
  int64_t fraction = tflite::IntegerFrExp(3.0, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(3.0, result, 1e-5);

  fraction = tflite::IntegerFrExp(123.45, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(123.45, result, 1e-5);

  fraction = tflite::IntegerFrExp(-23.232323, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(-23.232323, result, 1e-5);

  fraction = tflite::IntegerFrExp(static_cast<double>(NAN), &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_TRUE(std::isnan(result));

  fraction = tflite::IntegerFrExp(static_cast<double>(INFINITY), &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_FALSE(std::isfinite(result));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerDoubleMultiply) {
  TF_LITE_MICRO_EXPECT_NEAR(1.0, tflite::IntegerDoubleMultiply(1.0, 1.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, tflite::IntegerDoubleMultiply(1.0, 2.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, tflite::IntegerDoubleMultiply(2.0, 1.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(4.0, tflite::IntegerDoubleMultiply(2.0, 2.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(0.5, tflite::IntegerDoubleMultiply(1.0, 0.5), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(0.25, tflite::IntegerDoubleMultiply(0.5, 0.5),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(-1.0, tflite::IntegerDoubleMultiply(1.0, -1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(-1.0, tflite::IntegerDoubleMultiply(-1.0, 1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(1.0, tflite::IntegerDoubleMultiply(-1.0, -1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(
      15000000.0, tflite::IntegerDoubleMultiply(3000.0, 5000.0), 1e-5);
  TF_LITE_MICRO_EXPECT_TRUE(std::isnan(
      tflite::IntegerDoubleMultiply(static_cast<double>(NAN), 5000.0)));
  TF_LITE_MICRO_EXPECT_TRUE(std::isnan(
      tflite::IntegerDoubleMultiply(3000.0, static_cast<double>(NAN))));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerDoubleCompare) {
  TF_LITE_MICRO_EXPECT_EQ(-1, tflite::IntegerDoubleCompare(0.0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(1.0, 0.0));
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::IntegerDoubleCompare(1.0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::IntegerDoubleCompare(0.0, 0.0));
  TF_LITE_MICRO_EXPECT_EQ(-1, tflite::IntegerDoubleCompare(-10.0, 10.0));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(123.45, 10.0));
  TF_LITE_MICRO_EXPECT_EQ(
      1, tflite::IntegerDoubleCompare(static_cast<double>(NAN),
                                      static_cast<double>(INFINITY)));
  TF_LITE_MICRO_EXPECT_EQ(
      1, tflite::IntegerDoubleCompare(static_cast<double>(INFINITY),
                                      static_cast<double>(NAN)));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_PreprocessSoftmaxScaling) {
  auto quantize = [](double beta, double scale, int integer_bits) {
    int32_t q;
    int s;
    tflite::PreprocessSoftmaxScaling(beta, scale, integer_bits, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If beta * scale is greater than fits in the number of integer bits, the
  // result is move near the maximum. Otherwise they quantize as expected.
  // With 4 integer bits we can represent up to 16.0.

  auto r = quantize(1.0, 16.0, 4);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 2147483647);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  r = quantize(1.0, 8.0, 4);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 1073741824);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  // But with 5 bits we can go further.
  r = quantize(2.0, 16.0, 5);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 2147483647);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  r = quantize(2.0, 8.0, 5);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 1073741824);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_CalculateInputRadius) {
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(4, 27), 15);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(3, 27), 14);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(3, 28), 7);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(4, 2), 503316480);
}

TF_LITE_MICRO_TESTS_END

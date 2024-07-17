/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
// #include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
// #include "tensorflow/lite/micro/examples/hello_world/models/resnet_q_model_data.h"
// #include "tensorflow/lite/micro/examples/hello_world/models/mobilenetv3_q_tflite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
// #include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "models/conv_test_int8.h"
// #include "models/micro_speech_quantized_tflite.h"
// #include "models/resnet_q_tflite.h"

using HelloWorldOpResolver = tflite::MicroMutableOpResolver<100>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPadV2());
  TF_LITE_ENSURE_STATUS(op_resolver.AddHardSwish());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  return kTfLiteOk;
}

extern unsigned char resnet_q_tflite[];
TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(tf_micro_conv_test_model_int8_tflite);  // tf_micro_conv_test_model_int8_tflite g_hello_world_int8_model_data
  TFLITE_DCHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 300000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_DCHECK_NE(input, nullptr);

  for (uint32_t i=0; i<input->bytes/sizeof(float); i++)
    input->data.f[i] = i % 65535;

  interpreter.Invoke();
  
  TfLiteTensor* output = interpreter.output(0);
  TFLITE_DCHECK_NE(output, nullptr);

  for (uint32_t i=0; i<output->bytes/sizeof(float); i++)
    printf("%.6f, ", (double)output->data.f[i]);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  LoadQuantModelAndPerformInference();
  return kTfLiteOk;
}

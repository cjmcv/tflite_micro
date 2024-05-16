#include <stdio.h>
#include "py_tflite_inference.hpp"

int main() {
    printf("hello.\n");
    pai::infer::PyTfliteInference tflite_infer;
    tflite_infer.Init("tflite_infer", "TfliteInference", "../models/tf_micro_conv_test_model.int8.tflite");

    int8_t *input_data;
    uint32_t input_size;
    tflite_infer.GetInputPtr("serving_default_conv2d_input:0", (void **)&input_data, &input_size);

    for (uint32_t i=0; i<input_size/sizeof(uint8_t); i++)
        input_data[i] = i % 255;

    tflite_infer.Infer();

    tflite_infer.Print("StatefulPartitionedCall:0");

    return 0;
}
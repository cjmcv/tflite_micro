
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(1, os.path.jooin(CURRENT_PATH, "py_infer/"))

def infer(input_type):
    interpreter = tf.lite.Interpreter(model_path="../models/tf_micro_conv_test_model.int8.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    input=()
    for i in range(len(input_details)):
        print(input_details[i])
        if (input_type == np.int8):
            data = np.random.randint(0, 255, input_details[i]['shape'])
        else:
            data = np.random.rand(*input_details[i]['shape'])

        data = data.astype(input_type)
        interpreter.set_tensor(input_details[i]['index'], data)
        input = input + (data, )

    print("interpreter.invoke()")
    interpreter.invoke()

    output = ()
    for i in range(len(output_details)):
        print(output_details[i])
        o = interpreter.get_tensor(output_details[i]['index'])
        output = output + (o, )

    print("input: ", input)
    print("output: ", output)
    return input, output


if __name__ == "__main__":
    infer(np.int8) # np.int8 / np.float32

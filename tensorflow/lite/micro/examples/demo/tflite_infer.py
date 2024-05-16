import numpy as np
import tensorflow as tf

class TfliteInference:
    def hello(self):
        print("hello TfliteInference.")

    def load_model(self, model_path):
        print("TfliteInference->load_model:", model_path)
        self.interpreter = tf.lite.Interpreter(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()
        self.inputs = {}
        self.outputs = {}

        # print("\n")
        # for i in range(len(self.input_details)):
        #     print("in<", i, ">: ",self.input_details[i])
        # for i in range(len(self.output_details)):
        #     print("out<", i, ">: ",self.output_details[i])
        # print("\n")

    def get_io_num(self):
        return len(self.input_details), len(self.output_details)
    
    def get_inputs_detail(self, index):
        if(str(self.input_details[index]['dtype']) == "class 'numpy.float32>"):
            type_size = 4
        else:
            type_size = 1
        return type_size, self.input_details[index]['name'], self.input_details[index]['shape']
    
    def get_outputs_detail(self, index):
        if(str(self.output_details[index]['dtype']) == "class 'numpy.float32>"):
            type_size = 4
        else:
            type_size = 1
        return type_size, self.output_details[index]['name'], self.output_details[index]['shape']
    
    def set_input(self, index, data):
        self.inputs[index] = data
    
    def get_output(self, index):
        return self.outputs[index]
    
    def run(self):
        for i in range(len(self.input_details)):
            self.interpreter.set_tensor(self.input_details[i]['index'], self.inputs[i])

        self.interpreter.invoke()

        for i in range(len(self.output_details)):
            self.outputs[i] = self.interpreter.get_tensor(self.output_details[i]['index'])

    ################

    def fill_random_inputs(self):
        for i in range(len(self.input_details)):
            if(str(self.input_details[i]['dtype']) == "<class 'numpy.float32'>"):
                data = np.random.rand(*self.input_details[i]['shape'])
                data = data.astype(np.float32)
            else:
                data = np.random.randint(0, 255, self.input_details[i]['shape'])
                data = data.astype(np.int8)
            self.inputs[i] = data

        return self.inputs
    
    def get_npy_data(self, npy_data_file):
        data = np.load(npy_data_file)
        print(data.shape)
        print(data)
        return data
    
    ################

    def export_op(self):
        tensor_details  = self.interpreter.get_tensor_details()
        print(dir(self.interpreter))
        
        for tensor in tensor_details:
            print(tensor)
            i = tensor['index']
            tensor_name = tensor['name']
            scales = tensor['quantization_parameters']['scales']
            zero_points = tensor['quantization_parameters']['zero_points']
            tensor = self.interpreter.tensor(i)()

            print(i, tensor_name, scales.shape, zero_points.shape, tensor.shape)
    
if __name__ == "__main__":
    infer = TfliteInference()
    
    infer.load_model("../models/tf_micro_conv_test_model.int8.tflite")
    # infer.export_op()

    input = infer.fill_random_inputs()
    infer.run()
    output = infer.get_output(0)

    print(input)
    print(output)

            
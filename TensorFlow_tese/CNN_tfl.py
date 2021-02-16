import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10  
from sklearn.metrics import accuracy_score
import time

def model_lite():
    with tf.device('/cpu:0'):          #'/gpu:0' or '/TPU:0'
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        #size dataset, higth and hidth, color
        interpreter.resize_tensor_input(input_details[0]['index'], (10000, 32, 32, 3))
        interpreter.resize_tensor_input(output_details[0]['index'], (10000, 10))
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()

        print("input det", input_details[0]['shape'])
        print("input det", input_details[0]['dtype'])
        print("out det", output_details[0]['shape'])
        print("input det", output_details[0]['dtype'])

        start = time.time()

        interpreter.set_tensor(input_details[0]['index'], x_test)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        predictions_class = np.argmax(predictions, axis=1)

        acc = accuracy_score(predictions_class, y_test)
        end = time.time()
        print(acc)
        print(f"test time : {end-start}")

start = time
model_lite()
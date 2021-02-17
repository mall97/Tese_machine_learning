import os
import time

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"

import theano
import numpy as np
import keras
from keras import layers, regularizers
from keras.datasets import cifar10  
from keras.models import Sequential
from keras.utils import np_utils

#physical_devices = tf.config.list_physical_devices("GPU")
#print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


model = Sequential([
        layers.Conv2D(8, 3, padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(16, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(144, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),


        layers.Dense(100, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
    )

model2 = Sequential([
        layers.Conv2D(8, 3, padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(16, 3, padding="same"),      #mudar nós
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),


        layers.Dense(100, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
    )

model3 = Sequential([
        layers.Conv2D(8, 3, padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(16, 3, padding="same"),       #mudar nós
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),

        layers.Dense(100, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
    )

start = time.time()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('train')
model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=2)
end = time.time()
print('test')
model.evaluate(x_test, y_test, batch_size=10000, verbose=2)
end2 = time.time()

print(f"train time : {end-start}, test time : {end2-end}")

model.save("my_model")


#conda activate tf



#conda activate PythonGPU
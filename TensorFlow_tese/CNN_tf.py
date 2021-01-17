import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10  

physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="valid", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    # num of output, kernel size
    x = layers.Conv2D(8, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(144, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation="relu", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

with tf.device('/gpu:0'):
    model=my_model()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"],
    )

    print('train')
    model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
    print('test')
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)


model.save("my_model")


# Convert the model and Save the model.
converter = tf.lite.TFLiteConverter.from_saved_model("my_model") # path to the SavedModel directory
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

#conda activate tf
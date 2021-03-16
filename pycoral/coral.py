import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
from keras.datasets import cifar10  
from keras.utils import np_utils

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'tlite/1_20/model.tflite')
#label_file = os.path.join(script_dir, 'imagenet_labels.txt')
#image_file = os.path.join(script_dir, 'parrot.jpg')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

exit()

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
#image = Image.open(x_test).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
common.set_input(interpreter, x_test)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(y_test)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
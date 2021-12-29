import tensorflowjs
from tensorflow import keras

print(tensorflowjs.__version__)

# convert keras model to tensorflow js
model = keras.models.load_model('./dropout_02_380.h5')
tensorflowjs.converters.save_keras_model(model, './')

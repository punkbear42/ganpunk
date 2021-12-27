import tensorflowjs
from tensorflow import keras

print(tensorflowjs.__version__)

# convert keras model to tensorflow js
model = keras.models.load_model('./model-js/dropout_06_110.h5')
tensorflowjs.converters.save_keras_model(model, './')
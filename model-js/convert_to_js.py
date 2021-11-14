import tensorflowjs
from tensorflow import keras

print(tensorflowjs.__version__)

# convert keras model to tensorflow js
model = keras.models.load_model('./results/generator_model_e_158_b_1500.h5')
tensorflowjs.converters.save_keras_model(model, './')
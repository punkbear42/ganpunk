from tensorflow import keras
from helper import generate_fake_samples, save_punk, save_plot

def evaluate(latent_dim, n_samples):
	filename = 'results/generator_model_e_158_b_1500.h5'
	g_model = keras.models.load_model(filename)
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	save_punk(x_fake)

n_samples = 1
latent_dim = 100
evaluate(latent_dim, n_samples)
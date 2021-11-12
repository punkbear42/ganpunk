from tensorflow import keras
from helper import generate_similar_fake_samples, save_punk, save_plot

def evaluate(epoch, latent_dim, n_samples):
	filename = 'results/generator_model_e_1000_b_1000_fulloversample.h5'
	g_model = keras.models.load_model(filename)
	x_fake, y_fake = generate_similar_fake_samples(g_model, latent_dim, n_samples)
	save_plot(x_fake, 10, epoch)

n_samples = 100
latent_dim = 100
evaluate(485, latent_dim, n_samples)
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from numpy import tile
from matplotlib import pyplot
import os
import os

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n, epoch=-1):
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = 'results/generated_plot_e_%03d.png' % (epoch)
	pyplot.savefig(filename)
	pyplot.close()
	# TOKEN = os.getenv('TOKEN')
	# REPO = os.getenv('REPO')
	# if TOKEN != None:
	#	os.system("git add .")
	#	os.system("git commit --amend")
	#	os.system('git push https://' + TOKEN + '@github.com/' + REPO + '.git')

def save_punk(punk):
	# turn off axis
	# pyplot.axis('off')
	# plot raw pixel data
	# pyplot.imshow(punk)
	# print(punk)
	punk = (punk + 1) / 2.0
	filename = 'results/generated_punk.png'
	pyplot.imsave(filename, punk[0])
	# pyplot.close()

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

	# generate points in latent space as input for the generator
def generate_similar_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * 1)
	x_input = tile(x_input, n_samples)
	
	for x in range(100):
		x_input[x * 100] = x_input[x * 100] + x
		x_input[x * 99] = x_input[x * 99] + x
		x_input[x * 98] = x_input[x * 98] + x
	

	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	print(x_input)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)	
	
	y = zeros((n_samples, 1))
	return X, y

# use the generator to generate n fake examples, with class labels
def generate_similar_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_similar_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)	
	
	y = zeros((n_samples, 1))
	return X, y
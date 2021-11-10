from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from matplotlib import pyplot
import os
import os

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n, epoch=-1):
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, m, 1 + i)
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

def save_punk(punk, epoch, batch, n=10):
	# turn off axis
	# pyplot.axis('off')
	# plot raw pixel data
	# pyplot.imshow(punk)
	# print(punk)
	filename = 'results/generated_plot_e%03d_%03d.png' % (epoch+1, batch)
	pyplot.imsave(filename, punk)
	# pyplot.close()

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)	
	
	# print(X)
	# print ('after')
	# print(X)
	# print(X.shape)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y
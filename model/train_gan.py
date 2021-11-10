# example of training a gan on mnist
from numpy import expand_dims
from numpy import ones
from numpy import vstack
# from keras.datasets.mnist import load_data
import tensorflow as tf
print('version tensorflow')
print(tf.version.VERSION)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from helper import generate_fake_samples, generate_latent_points, save_plot
from imblearn.over_sampling import RandomOverSampler
import os

img_height = 24
img_width = 24
batch_size = 2000

# define the standalone discriminator model
def define_discriminator(in_shape=(img_height,img_width,4)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 128 * 6 * 6
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((6, 6, 128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # 12x12
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # 24x24
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(4, (3,3), activation='tanh', padding='same'))
	model.summary()
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def show_dataset(train_data):
	# print(train_data)
	fig = pyplot.figure(figsize=(10, 10))
	fig.patch.set_facecolor('none')
	fig.patch.set_alpha(0.0)
	incr = 0
	for x in train_data:
    	# pyplot.subplot(5, 5, 1 + incr)
		# print(x.shape)
		# print(x[0])
		pyplot.imshow(x[0].numpy().astype("uint"))
		# pyplot.title(class_names[labels[i]])
		pyplot.axis("off")
		pyplot.show()
		incr = incr + 1
		if incr > 5:
			break
	
		
def load_punks():
	train_data = image_dataset_from_directory(
		'./punks-classified',
		labels='inferred',
		label_mode='int',
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		color_mode='rgba')
		
	# show_dataset(train_data)
	return train_data
	
# load and prepare mnist training images
def load_real_samples():
	trainX = load_punks()
		
	# trainX = trainX.map(lambda x: (x - 127.5) / 127.5)
	
	return trainX

# select real samples
def generate_real_samples(dataset, n_samples):
	y = ones((n_samples, 1))
	return dataset, y

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, batch, g_model, d_model, dataset, latent_dim, n_samples=batch_size):
	# prepare real samples

	X_real, y_real = generate_real_samples(dataset[0], dataset[0].shape[0])
	# X_real, y_real = generate_real_samples(dataset, n_samples)

	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, 10, epoch)
	# save the generator model tile file
	filename = 'results/generator_model_e_%03d.h5' % (epoch))
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=batch_size):
	# bat_per_epo = int(dataset.shape[0] / n_batch)
	incr = 0
	# manually enumerate epochs
	for current_epoch in range(n_epochs):
		# enumerate batches over the training set
		current_batch = 0
		lastDataset = 0
		for j in dataset:
			
			# smote
			sm = RandomOverSampler(random_state=42)
			reshaped = j[0].numpy().reshape(j[0].numpy().shape[0], img_height * img_width * 4)
			X_real,_ = sm.fit_resample(reshaped, j[1])
			X_real = X_real.reshape(-1, img_height, img_width, 4)
			# normalize
			X_real = (X_real - 127.5) / 127.5 # -1, 1
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, batch_size)

			_, y_real_ = generate_real_samples(j[0], X_real.shape[0])
			# _, y_real_ = generate_real_samples(j, batch_size)

			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real_, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('epoch=%d, batch=%d, d=%.3f, g=%.3f' % (current_epoch+1, current_batch+1, d_loss, g_loss))
			current_batch = current_batch + 1
			lastDataset = j

		if current_epoch % 2 == 0:
				summarize_performance(current_epoch, current_batch, g_model, d_model, lastDataset, latent_dim)
			

		
				
			
		
# evaluate the model performance, sometimes
# TOKEN = os.getenv('TOKEN')
# REPO = os.getenv('REPO')
# if TOKEN != None:
#	os.system("git checkout -b bot")

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
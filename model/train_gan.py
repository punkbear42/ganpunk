from numpy import ones
from matplotlib import pyplot
import argparse
import numpy
import tensorflow as tf
print('version tensorflow')
print(tf.version.VERSION)
from numpy.random import rand
from numpy.random import randint
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

from imblearn.over_sampling import RandomOverSampler

from helper import generate_fake_samples, generate_latent_points, save_plot


IMG_HEIGHT = 24
IMG_WIDTH = 24


def define_discriminator(in_shape=(IMG_HEIGHT, IMG_WIDTH, 4),
                         lr=0.0002, beta_1=0.5, dropout=0.4):
    """Define the standalone discriminator model."""
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=lr, beta_1=beta_1)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def define_generator(latent_dim):
    """Define the standalone generator model."""
    model = Sequential()
    n_nodes = 256 * 3 * 3
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((3, 3, 256)))
    model.add(Conv2DTranspose(
        128, (3, 3), strides=(2, 2), padding='same'))  # 6x6
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(
        128, (3, 3), strides=(2, 2), padding='same'))  # 12x12
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding='same'))  # 24x24
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(4, (3, 3), activation='tanh', padding='same'))
    model.summary()
    return model


def define_gan(g_model, d_model, lr=0.0002, beta_1=0.5):
    """Define the combined generator and discriminator model, for updating the
        generator."""
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=lr, beta_1=beta_1)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def show_dataset(train_data):
    fig = pyplot.figure(figsize=(10, 10))
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    incr = 0
    for x in train_data:
        pyplot.imshow(x[0].numpy().astype("uint"))
        pyplot.axis("off")
        pyplot.show()
        incr = incr + 1
        if incr > 5:
            break


def load_punks(batch_size, data_sampling=""):
    if data_sampling == "":
        # raw 10.000 are being loaded
        train_data = image_dataset_from_directory(
            './punks',
            label_mode=None,
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=10000,
            color_mode='rgba')
        train_data = next(iter(train_data)).numpy()

    elif data_sampling == "BasicClassifier":
        # classified punks are being loaded,
        # punks are duplicated over multiple classes
        # e.g a single punk can be found in the class "bear" and in the class "beanie".
        train_data = image_dataset_from_directory(
            './punks-classified',
            labels='inferred',
            label_mode='int',
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=30000000, # very high number so everything get into one batch
            color_mode='rgba')
        train_data = next(iter(train_data))
        train_data = train_data[0].numpy()

    elif data_sampling == "RandomOverSampler":
        # same as BasicClassifier
        # + RandomOverSampler: e.g it fixes imbalanced class (e.g if the class "bear" and "beanie" are imbalanced)
        train_data = image_dataset_from_directory(
            './punks-classified',
            labels='inferred',
            label_mode='int',
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=30000000, # very high number so everything get into one batch
            color_mode='rgba')
        train_data = next(iter(train_data))

        data_punks = train_data[0].numpy()
        label_punks = train_data[1].numpy()

        sm = RandomOverSampler(random_state=42)

        reshaped = data_punks.reshape(
            data_punks.shape[0], IMG_HEIGHT * IMG_WIDTH * 4)
        train_data, _ = sm.fit_resample(reshaped, label_punks)
        train_data = train_data.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 4)
    
    print(f'samples: {train_data.shape[0]}')

    return train_data

def generate_real_samples(dataset, n_samples):
	# choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    
    # retrieve selected images
    X = dataset[ix]

	# generate 'real' class labels (1)

	
    y = ones((n_samples, 1))
	
    return X, y

def summarize_performance(output_file, epoch, d_loss_real, d_loss_fake, g_loss, acc_real, acc_fake,
                          g_model, d_model, dataset, latent_dim, n_samples,
                          save_model=False):
    # prepare real samples
    # X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    # _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    # _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('Accuracy real: %.0f%%, fake: %.0f%%' %
          (acc_real * 100, acc_fake * 100))

    d_loss = (d_loss_real + d_loss_fake) / 2
    with open(f'{output_file}_training.txt', 'a') as file:
        file.write(
            f'{epoch},{d_loss},{d_loss_real},{d_loss_fake},{g_loss},{acc_real},{acc_fake}\n')

    if save_model:
        # save plot
        save_plot(x_fake, 10, epoch, output_file)
        # save the generator model tile file
        filename = f'{output_file}_{epoch}.h5'
        g_model.save(filename)


def train(output_file, g_model, d_model, gan_model, dataset, latent_dim,
          n_epochs, batch_size, checkpoint_every_epochs=50):
    """Train the generator and discriminator."""
    
    with open(f'{output_file}_training.txt', 'w') as file:
        file.write(
            'Epoch,Discriminator loss,Discriminator loss (Real),Discriminator loss (Fake),Generator loss,Accuracy real,Accuracy fake\n')

    half_batch = int(batch_size / 2)
    batch_per_epoch = int(dataset.shape[0] / batch_size)

    for current_epoch in range(n_epochs):
        # enumerate batches over the training set
                
        for current_batch in range(batch_per_epoch):

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # normalize
            X_real = (X_real - 127.5) / 127.5  # -1, 1
        

            # update discriminator model weights
            d_loss_real, acc_real = d_model.train_on_batch(X_real, y_real)
            d_loss_fake, acc_fake = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, batch_size)
            # create inverted labels for the fake samples
            y_gan = ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('epoch=%d, batch=%d, d_real=%.3f, d_fake=%.3f, g=%.3f' %
                (current_epoch + 1, current_batch + 1, d_loss_real,
                d_loss_fake, g_loss))
        
        summarize_performance(output_file, current_epoch,
                              d_loss_real, d_loss_fake,
                              g_loss, acc_real, acc_fake, g_model, d_model, dataset,
                              latent_dim, batch_size,
                              current_epoch % checkpoint_every_epochs == 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True,
                        help="Will save the model into a h5 file with the "
                             "name provided. E.g. 'experiment1' will generate"
                             " files experiment1.h5, "
                             "experiment1_training.txt with the losses and "
                             "classification performance.")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--beta_1", default=0.5, type=float)
    parser.add_argument("--latent_dimensions", type=int, default=100)
    parser.add_argument("--checkpoint_every_epochs", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--data_sampling", type=str, default="", choices=['RandomOverSampler', 'BasicClassifier'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # create the discriminator
    d_model = define_discriminator(lr=args.learning_rate,
                                   beta_1=args.beta_1)
    # create the generator
    g_model = define_generator(args.latent_dimensions)
    # create the gan
    gan_model = define_gan(g_model, d_model, lr=args.learning_rate,
                           beta_1=args.beta_1)

    # load image data
    dataset = load_punks(args.batch_size, args.data_sampling)
    # train model
    train(args.output_file, g_model, d_model, gan_model, dataset,
          args.latent_dimensions, args.n_epochs, args.batch_size,
          args.checkpoint_every_epochs)

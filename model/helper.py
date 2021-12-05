from numpy.random import randn
from numpy import zeros
from numpy import tile
from matplotlib import pyplot

from DiffAugment_tf import DiffAugment

def save_plot(examples, n, epoch=-1, base_file_name='results/generated_plot'):
    """Create and save a plot of generated images (reversed grayscale)"""
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
    filename = f'{base_file_name}_e_%03d.png' % (epoch)
    pyplot.savefig(filename)
    pyplot.close()


def save_punk(punk, filename='results/generated_punk.png'):
    punk = (punk + 1) / 2.0
    pyplot.imsave(filename, punk[0])


def generate_latent_points(latent_dim, n_samples):
    """Generate points in latent space as input for the generator."""
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_similar_latent_points(latent_dim, n_samples):
    """Generate points in latent space as input for the generator."""
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


def generate_fake_samples(g_model, latent_dim, n_samples):
    """Use the generator to generate n fake examples, with class labels."""
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    x_input = DiffAugment(x_input)
    # predict outputs
    X = g_model.predict(x_input)

    y = zeros((n_samples, 1))
    return X, y


def generate_similar_fake_samples(g_model, latent_dim, n_samples):
    """Use the generator to generate n fake examples, with class labels."""
    # generate points in latent space
    x_input = generate_similar_latent_points(latent_dim, n_samples)
    x_input = DiffAugment(x_input)
    # predict outputs
    X = g_model.predict(x_input)

    y = zeros((n_samples, 1))
    return X, y

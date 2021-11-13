import os
import argparse

from tensorflow import keras

from helper import generate_fake_samples, save_punk
import shutil


"""
Command line tool to generate punks from a pre-trained model.
You can provide the pre-trained model file, the number of punks to generate,
the number of dimensions from the latent space, etc.
Each punk will be saved as a separate file in the specified output folder.

@author pere
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", required=True,
                        help="Output folder to save generated punks.")
    parser.add_argument("--n_punks", default=1, type=int,
                        help="Number of punks to generate (each will be "
                             "saved in a different file.")
    parser.add_argument("--model_name",
                        default="results/generator_model_e_158_b_1500.h5",
                        help="Trained model file to use (h5)")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="# latent dimensions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    g_model = keras.models.load_model(args.model_name)
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)

    os.makedirs(args.output_folder)

    for i in range(0, args.n_punks):
        n_samples = 1
        x_fake, _ = generate_fake_samples(g_model, args.latent_dim, 1)
        save_punk(x_fake, filename=os.path.join(
            args.output_folder, f'result_{i}.png'))

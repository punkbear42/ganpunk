import sys

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception('plot_training.py <training_file.txt>')
    training_file = sys.argv[1]

    file = pd.read_csv(training_file)
    plot = file.plot.line(y=['Discriminator loss', 'Generator loss'])

    plt.show()

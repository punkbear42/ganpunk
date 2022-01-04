import psutil
import time
import glob
import os

from PIL import Image

"""
Interactive display of generated punks using PIL

@author pere
"""

if __name__ == '__main__':
    punks_folder = 'model_oversample'

    for punk_file in glob.glob(os.path.join(punks_folder, '*png')):
        img = Image.open(punk_file)

        new_image = Image.new("RGBA", img.size, "WHITE")
        new_image.paste(img, (0, 0), img)

        img = new_image.convert('RGB')

        image = img.convert('P', palette=Image.ADAPTIVE)\
            .resize((256, 256))

        image.show()

        time.sleep(2)

        # hide image (super hacky way)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()

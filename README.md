# ganpunk

# splitting punks

./magick convert /home/punkbear/punk/punks.png -crop 24x24  +repage  +adjoin  punk_%02d.png

# deps

pip install imblearn
pip install matplotlib

# training

python model/train_gan.py --output_file test --batch_size 128 --checkpoint_every_epochs 1 --n_epochs 50

# How to use data augmentation

The code will automatically apply DiffAugment if present in the path.

```
git clone https://github.com/mit-han-lab/data-efficient-gans.git
```

```
import sys
sys.path.insert(1, "./data-efficient-gans/DiffAugment-stylegan2/")
```



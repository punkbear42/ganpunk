# ganpunk

# splitting punks

./magick convert /home/punkbear/punk/punks.png -crop 24x24  +repage  +adjoin  punk_%02d.png

# deps

pip install imblearn
pip install matplotlib

# training

python model/train_gan.py --output_file test --batch_size 256 --checkpoint_every_epochs 3 --n_epochs 150 --data_sampling BasicClassifier



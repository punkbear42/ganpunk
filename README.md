# ganpunk

# splitting punks

./magick convert /home/punkbear/punk/punks.png -crop 24x24  +repage  +adjoin  punk_%02d.png

# deps

pip install imblearn
pip install matplotlib

# training

python model/train_gan --output_file run-18-11 --batch_size 128 --checkpoint_every_epochs 10 --n_epochs 200



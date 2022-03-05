import tensorflow as tf
from train import train

img_height = 64
img_width = 64
batch_size = 128
EPOCHS = 250

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode=None)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale= 1./127.5, offset=-1)
train_ds = train_ds.map(lambda x: normalization_layer(x))

trainGAN = train(train_ds, EPOCHS)
trainGAN.trainGAN()     
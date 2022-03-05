from dis import dis
from operator import ge
from this import d
import tensorflow as tf
from tensorflow.keras import layers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class models():


    def __init__(self):
        pass

    def make_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(1, 1, 100)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Reshape((4, 4, 1024)))

        model.add(layers.Conv2DTranspose(
            512, 5, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(
            256, 5, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(
            128, 5, strides=2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(3, 5, strides=2,
                  padding='same', use_bias=False, activation='tanh'))

        return model

    def make_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Reshape((64, 64, 3)))

        model.add(layers.Conv2D(64, 5, 2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, 5, 2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(256, 5, 2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(512, 5, 2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
    
    def generator_loss(self,fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def disriminator_loss(self,real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_optimizer(self):
        optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        return optimizer

    def discriminator_optimizer(self):
        optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        return optimizer

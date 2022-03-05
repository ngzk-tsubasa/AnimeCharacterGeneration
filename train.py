import tensorflow as tf
from tensorflow.keras import layers 
from model import models
import matplotlib.pyplot as plt
import time 
import cv2

noise_dim = 100
batch_size = 128
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate,1 ,1, noise_dim])

class train():

    def __init__(self, dataset, epochs):
        self.models = models()
        self.generator = self.models.make_generator()
        self.discriminator = self.models.make_discriminator()
        self.generator_optimizer = self.models.generator_optimizer()
        self.discriminator_optimizer = self.models.discriminator_optimizer()
        self.epochs = epochs
        self.dataset = dataset

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([batch_size,1 ,1 ,noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.models.generator_loss(fake_output)
            disc_loss = self.models.disriminator_loss(real_output, fake_output)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def generate_and_save_image(self, model, epoch, test_input):
        predictions = model(test_input, training = False)
        
        
        fig = plt.figure(figsize = (4,4))
        
        for i in range(predictions.shape[0]):
            
            r = predictions[i, :, :, 0]
            g = predictions[i, :, :, 1]
            b = predictions[i, :, :, 2]
            
            r = r.numpy()
            g = g.numpy()
            b = b.numpy()
            
            merged = cv2.merge([b,g,r])
            merged = cv2.normalize(merged, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            plt.subplot(4, 4, i+1)
            plt.imshow(merged/255)
            plt.axis('off')
           

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.close('all')
        plt.show()


    def trainGAN(self):
        for epoch in range(self.epochs):
            start = time.time()
        
            for image_batch in self.dataset:
                self.train_step(image_batch)

            #display.clear_output(wait=True)
            self.generate_and_save_image(self.generator,
                                    epoch + 1,
                                    seed)

            

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        #display.clear_output(wait=True)
        self.generate_and_save_image(self.generator,
                                self.epochs,
                                seed)
    
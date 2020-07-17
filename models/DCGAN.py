import tensorflow as tf

from .BaseGAN import BaseGAN
from .types_ import *
from .utils import makeLayers

tfk = tf.keras
tfkl = tf.keras.layers

class DCGAN(BaseGAN):
    def __init__(self, 
                model_name: str = 'DCGAN',
                latent_dim: int = 100,
                latent_batch_size: int = 32,
                discriminate_param: dict = None,
                generate_param: dict = None) -> None:
        super(DCGAN, self).__init__()
        
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.latent_batch_size = latent_batch_size

        # --- Generate
        generate_input = tfkl.Input(shape=generate_param['input_shape'])
        for index, layer_spec in enumerate(generate_param['layers']):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(generate_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.generator = tfk.Model(inputs= generate_input, outputs=x)

        # --- Discriminate
        discriminate_input = tfkl.Input(shape=discriminate_param['input_shape'])
        for index, layer_spec in enumerate(discriminate_param['layers']):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(discriminate_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.discriminator = tfk.Model(inputs= discriminate_input, outputs=x)

    def generate(self, z: Tensor=None) -> Tensor:
        if z is None :
            z = tf.random.normal(shape= (1,self.latent_dim))
        return self.generator(z)

    def discriminate(self, x: Tensor=None) -> Tensor:
        return self.discriminator(x)

    def generate_loss(self, y_hat:Tensor=None) -> Tensor:
        return tfk.losses.BinaryCrossentropy()(tf.ones_like(y_hat), y_hat)

    def discriminate_loss(self, y:Tensor, y_hat:Tensor) -> Tensor:
        real_loss = tfk.losses.BinaryCrossentropy()(tf.ones_like(y), y)
        fake_loss = tfk.losses.BinaryCrossentropy()(tf.zeros_like(y_hat), y_hat)
        total_loss = real_loss + fake_loss
        return total_loss

    def compute_loss(self, x=None, z:Tensor = None):
        if z is None :
            z = tf.random.normal([self.latent_batch_size, self.latent_dim])
        
        x_hat = self.generate(z)
        if x is not None:
            y = self.discriminate(x)
        y_hat = self.discriminate(x_hat)

        gen_loss = self.generate_loss(y_hat)
        if x is not None:
            disc_loss = self.discriminate_loss(y, y_hat)
        else : disc_loss = None

        return gen_loss, disc_loss

    def train_step_disc(self, 
                        x:Tensor= None, 
                        disc_opt= tfk.optimizers.Adam(1e-4)) -> Tensor:

        with tf.GradientTape() as disc_tape:
            _, disc_loss = self.compute_loss(x)

        gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
        return

    def train_step_gen(self, 
                        z:Tensor= None, 
                        gen_opt= tfk.optimizers.Adam(1e-4)) -> Tensor:

        with tf.GradientTape() as gen_tape:
            gen_loss, _ = self.compute_loss(x=None, z=z)

        gradient_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
        return

    @tf.function
    def sample(self, z:Tensor = None) -> Tensor:
        return self.generate(z)


        

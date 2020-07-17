import tensorflow as tf

from .BaseGAN import BaseGAN
from .types_ import *
from .utils import makeLayers

tfk = tf.keras
tfkl = tf.keras.layers

class BiGAN(BaseGAN):
    def __init__(self, 
                model_name: str = 'BiGAN',
                latent_dim: int = 100,
                latent_batch_size: int = 32,
                encode_params: dict = None,
                generate_params: dict = None) -> None:
        super(BiGAN, self).__init__()

        self.model_name = model_name
        self.latent_dim = latent_dim
        self.latent_batch_size = latent_batch_size

        # --- Generate
        generate_input = tfkl.Input(shape=generate_params['input_shape'])
        for index, layer_spec in enumerate(generate_params['layers']):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(generate_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.generator = tfk.Model(inputs=generate_input, outputs= x)

        # --- Encode
        encode_input = tfkl.Input(shape=encode_params['input_shape'])
        for index, layer_spec in enumerate(encode_params['layers']):
            if index is 0 :
                x = makeLayers(layer_spec=layer_spec)(encode_input)
            else :
                x = makeLayers(layer_spec=layer_spec)(x)
        self.encoder = tfk.Model(inputs=encode_input, outputs=x)

        # --- discriminate
        discriminate_x_input = tfkl.Input(shape=(128, 128, 3))
        discriminate_z_input = tfkl.Input(shape=(100,))

        x = tfkl.Conv2D(filters=32, kernel_size=7, strides=(4, 4), padding='same')(discriminate_x_input)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Flatten()(x)

        x = tfkl.concatenate([x, discriminate_z_input])
        x = tfkl.Dense(10000)(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU()(x)

        x = tfkl.Dense(1000)(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU()(x)

        x = tfkl.Dense(1, activation='sigmoid')(x)

        self.discriminator = tfk.Model(inputs = [discriminate_x_input, discriminate_z_input], outputs = x)


    def generate(self, z:Tensor=None) -> Tensor:
        if z is None :
            z = tf.random.normal(shape=(1, self.latent_dim))
        return self.generator(z)
        

    def discriminate(self, x:Tensor=None, z:Tensor=None) -> Tensor:
        if x is not None and z is None:
            # real_data case
            z = self.encode(x)

        if x is None and z is not None:
            x = self.generate(z)
        
        return self.discriminator([x, z])

    def encode(self, x:Tensor) -> Tensor:
        return self.encoder(x)

    def generate_loss(self, y_hat:Tensor) -> Tensor:
        return tfk.losses.BinaryCrossentropy()(tf.ones_like(y_hat), y_hat)

    def encode_loss(self, y:Tensor):
        return tfk.losses.BinaryCrossentropy()(tf.ones_like(y), y)

    def discriminate_loss(self, y:Tensor, y_hat:Tensor) -> Tensor:
        real_loss = tfk.losses.BinaryCrossentropy()(tf.ones_like(y), y)
        fake_loss = tfk.losses.BinaryCrossentropy()(tf.zeros_like(y_hat), y_hat)
        return 0.5*(real_loss+fake_loss)

    def compute_loss(self, x:Tensor=None, z:Tensor=None) -> Tuple[Gen_loss, Disc_loss]:
        if z is None :
            z = tf.random.normal([self.latent_batch_size, self.latent_dim])

        x_hat = self.generate(z)
        if x is not None:
            z_hat = self.encode(x)
            y = self.discriminate(x, z_hat)

        y_hat = self.discriminate(x_hat, z)

        gen_loss = self.generate_loss(y_hat)
        if x is not None:
            disc_loss = self.discriminate_loss(y, y_hat)
            en_loss = self.encode_loss(y)
        else :
            disc_loss = None
            en_loss = None
        return gen_loss, disc_loss, en_loss

    @tf.function
    def train_step_gen(self, 
                        z:Tensor=None, 
                        gen_opt: Optimizer=tfk.optimizers.Adam(1e-4)) -> Tensor:
        with tf.GradientTape() as gen_tape:
            gen_loss, _, _ = self.compute_loss(x=None, z=z)
        gradient_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
        return 

    @tf.function
    def train_step_disc(self, 
                        x:Tensor=None,  
                        disc_opt= tfk.optimizers.Adam(1e-4),
                        en_opt=tfk.optimizers.Adam(1e-4)) -> Tensor:
        with tf.GradientTape() as disc_tape, tf.GradientTape() as en_tape:
            _, disc_loss, en_loss = self.compute_loss(x)

        gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))

        gradient_en = en_tape.gradient(en_loss, self.encoder.trainable_variables)
        en_opt.apply_gradients(zip(gradient_en, self.encoder.trainable_variables))
        return 

    @tf.function
    def sample(self, z:Tensor=None) -> Tensor:
        return self.generate(z)
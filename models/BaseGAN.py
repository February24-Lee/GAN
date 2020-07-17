import tensorflow as tf
from abc import abstractmethod

from .types_ import *


tfk = tf.keras
tfkl = tf.keras.layers

class BaseGAN(tfk.Model):

    def __init__(self):
        super(BaseGAN, self).__init__()

    def generate(self, z:Tensor) -> Tensor:
        raise NotImplementedError

    def discriminate(self, x:Tensor) -> Tensor:
        raise NotImplementedError

    def generate_loss(self, y_hat:Tensor) -> Tensor:
        raise NotImplementedError

    def discriminate_loss(self, y:Tensor, y_hat:Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, x:Tensor) -> List[Gen_loss, Disc_loss]:
        pass

    @abstractmethod
    def train_step_gen(self, z:Tensor, gen_opt: Optimizer) -> Tensor:
        pass

    @abstractmethod
    def train_step_disc(self, x:Tensor,  disc_opt: Optimizer) -> Tensor:
        pass

    @abstractmethod
    def sample(self, z:Tensor) -> Tensor:
        pass

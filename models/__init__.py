from .DCGAN import DCGAN
from .BiGAN import BiGAN
from .trainer import *
from .types_ import *
from .utils import *

GAN = DCGAN

gan_models = {
    "DCGAN" : DCGAN,
    "BiGAN" : BiGAN
}
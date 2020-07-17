import yaml
import argparse
import numpy as np

from models import *
from utils.dataset_CelebA import genDatasetCelebA
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    default=None)

parser.add_argument('--save_model', '-s',
                    dest="Is_save_model",
                    default='true')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try :
        config = yaml.load(file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(e)

# --- dataset
train_gen, test_gen = genDatasetCelebA(**config['dataset_params'])

# --- model
model = gan_models[config['model_params']['model_name']](**config['model_params'])

# --- Optimizer
if config['opt_params']['gen_opt']['name'] is 'Adam':
    gen_opt = tfk.optimizers.Adam(config['opt_params']['gen_opt']['learning_rate'])

if config['opt_params']['disc_opt']['name'] is 'Adam':
    disc_opt = tfk.optimizers.Adam(config['opt_params']['disc_opt']['learning_rate'])

# --- train
trainer(model, 
        train_gen, 
        test_gen, 
        gen_opt= gen_opt,
        disc_opt= disc_opt,
        epochs=config['train_params']['epochs'],
        iter_disc=config['train_params']['iter_disc'],
        iter_gen=config['train_params']['iter_gen'],
        save_path=config['train_params']['save_path'],
        save_model_path = config['train_params']['save_model_path'],
        scale=config['dataset_params']['scale'],
        batch_size=config['dataset_params']['batch_size'])

# --- save model
if args.Is_save_model == 'true':
    path = config['train_params']['save_model_path'] + model.model_name +'.h5'
    model.save_weights(path)
     

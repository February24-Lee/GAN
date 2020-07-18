import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .types_ import *

tfk = tf.keras
tfkl = tf.keras.layers

def trainer(model,
            train_x: DirectoryIterator = None,
            test_x: DirectoryIterator =None,
            gen_opt=tfk.optimizers.Adam(1e-4),
            disc_opt=tfk.optimizers.Adam(1e-4),
            en_opt=tfk.optimizers.Adam(1e-4),
            epochs=10,
            iter_disc= 1,
            iter_gen= 1,
            save_path: str= None,
            save_model_path: str= None,
            scale='sigmoid',
            batch_size:int =32):

    train_iter = train_x.n // batch_size 
    test_iter = test_x.n // batch_size

    for epoch in range(1, epochs+1):
        start_t = time.time()
        print('Epoch : {} training..'.format(epoch))
        for index in tqdm(range(train_iter)):
            # --- train for disc
            for _ in range(iter_disc):
                x = next(train_x)
                if index > train_iter:
                    break

                if scale == 'tanh':
                    x = (x-127.5)/127.5 
                elif scale == 'sigmoid':
                    x = x/255.

                model.train_step_disc(x, disc_opt=disc_opt)

            # --- train for gen
            for _ in range(iter_gen):
                model.train_step_gen(gen_opt=gen_opt)

        end_time = time.time()

        print('Calculating testset...')
        gen_loss = tfk.metrics.Mean()
        disc_loss = tfk.metrics.Mean()
        en_loss = tfk.metrics.Mean()
        for index, x in enumerate(test_x):
            if index > test_iter:
                break
            if scale == 'tanh':
                x = (x-127.5)/127.5 
            elif scale == 'sigmoid':
                x = x/255.

            if hasattr(model,'encode'):
                gen_test_loss, disc_test_loss, en_test_loss =  model.compute_loss(x)
                en_loss(en_test_loss)
            else:    
                gen_test_loss, disc_test_loss =  model.compute_loss(x)
            gen_loss(gen_test_loss)
            disc_loss(disc_test_loss)
        if hasattr(model, 'encode'):
            gen_mean_loss = -gen_loss.result()
            disc_mean_loss = -disc_loss.result()
            en_mean_loss = -en_loss.result()
            print('Epoch: {}, gen loss: {}, disc loss:{}, en_loss:{} time elapse for current epoch: {}'
              .format(epoch, gen_mean_loss, disc_mean_loss, en_mean_loss, end_time - start_t))
        else:
            gen_mean_loss = -gen_loss.result()
            disc_mean_loss = -disc_loss.result()
            print('Epoch: {}, gen loss: {}, disc loss:{} time elapse for current epoch: {}'
              .format(epoch, gen_mean_loss, disc_mean_loss, end_time - start_t))
        path = save_path + model.model_name + '_epoch_' + str(epoch) + '.png'
        save_sample_images(model, img_num=64, path=path)
    return 

def save_sample_images(model,
                img_num=64,
                path: str=None):
    z = tf.random.normal([img_num, model.latent_dim])
    sample_img = model.sample(z)
    plt.figure(figsize=(15,15))
    for i in range(img_num):
        plt.subplot(8,8,i+1)
        plt.imshow(sample_img[i])
        plt.axis('off')
    plt.savefig(path)
    return




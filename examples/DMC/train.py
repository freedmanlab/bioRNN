import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from models import bioRNN_Model
from stimulus2 import Stimulus
from parameters2 import par
import argparse
import time


parser = argparse.ArgumentParser(description='Supervised trianing of bioRNN \
        with default params on DMC task..')
parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use')

if __name__=='__main__':
    # Setup
    args = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')

    # Get model
    model = bioRNN_Model(par['n_input'], par['n_hidden'], par['n_output'])

    # Train model
    stim = Stimulus()
    opt = tf.keras.optimizers.Adam(learning_rate=par['learning_rate'])
    for i in range(par['num_iterations']):
        start = time.time()
        trial_info = stim.generate_trial()
        input_data = tf.constant(trial_info['neural_input'], dtype=tf.float32)
        ytrue_data = tf.constant(trial_info['desired_output'], dtype=tf.float32)
        dead_time_mask = tf.constant(trial_info['train_mask'], dtype=tf.float32)
        data = ((input_data, dead_time_mask), ytrue_data)
        metrics = model.train_step(opt, data)
        end = time.time()
        if i%100==0:
            loss = round(float(metrics['loss']), 2)
            spike_loss = round(float(metrics['spike_cost']), 2)
            weight_loss = round(float(metrics['weight_cost']), 2)
            acc = round(float(metrics['accuracy']), 2)
            etime = round(end - start, 2)
            tf.print(f'Iter: {i} | Loss: {loss} | Acc: {acc} | Spike: {spike_loss} | Weight: {weight_loss} | Time (s/iter): {etime} \n')

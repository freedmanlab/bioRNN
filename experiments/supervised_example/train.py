import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ori any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
from model import bioRNN
from training import supervised_train_step
from xdg_stimulus import MultiStimulus
from utils import read_config, save_config


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use')
parser.add_argument('--config', type=str, default='./config.yaml',
    help='Directory of config file.')

if __name__=='__main__':
    # Setup
    args = parser.parse_args()
    config = read_config(args.config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    stim = MultiStimulus()

    # Define model
    rnn = bioRNN(config['hidden_size'], config['output_size'],
        synapse_config=config['synapse_config'])

    # If you want to save this run, check savedir is legit and prep cpoints
    if config['save'] is True:
        sd = config['savedir']
        if not os.path.exists(sd):
            os.makedirs(sd)
        cpoint = tf.train.Checkpoint(module=rnn)
        config_fn = os.path.join(sd, 'config.yaml')
        save_config(config, config_fn)

    # Train
    opt = tf.keras.optimizers.Adam(learning_rate=float(config['learning_rate']))
    loss_list = []
    acc_list = []
    print('\n=== TRAINING === \n')
    for i in range(config['n_iterations']):
        name, inputs, labels, mask, rewards = stim.generate_trial(config['task_idx'])
        metrics = supervised_train_step(rnn, opt, inputs, labels, train_mask=mask)
        loss = round(float(metrics['loss']), 2)
        acc = round(float(metrics['argmax_accuracy']), 2)
        dec_acc = round(float(metrics['decision_accuracy']), 2)
        fix_acc = round(float(metrics['fixation_accuracy']), 2)
        loss_list.append(loss)
        acc_list.append(acc)
        if i%100==0:
            print(f'Iter: {i} | Loss: {loss} | Acc: {acc} | Fixation Acc: {fix_acc} | Decision Acc: {dec_acc}')

    # Save model
    if config['save'] is True:
        cpoint_fn = os.path.join(sd, 'cpoint')
        cpoint.save(cpoint_fn)
        acc_fn = os.path.join(sd, 'accs.npy')
        np.save(acc_fn, acc_list)
        loss_fn = os.path.join(sd, 'loss.npy')
        np.save(loss_fn, loss_list)

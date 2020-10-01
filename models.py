import tensorflow as tf
import numpy as np
from layers import bioRNN_Cell
import initializers
import losses
import metrics

class bioRNN_Model(tf.keras.Model):
    def __init__(self, n_input, n_hidden, n_output, EI=True, excitatory_frac=0.8,
        balance_EI=True, connection_prob=1., synapse_config='full', n_receptive_fields=1.,
        dt=10, tau_slow=1500., tau_fast=200., membrane_time_constant=100.,
        noise_rnn_sd=0.5, structure_mask=None, **kwargs):

        super().__init__(**kwargs)

        # Initialize RNN
        self.rnn = bioRNN_Cell(n_hidden,
            EI=EI,
            excitatory_frac=excitatory_frac,
            balance_EI=balance_EI,
            connection_prob=connection_prob,
            synapse_config=synapse_config,
            n_receptive_fields=n_receptive_fields,
            dt=dt,
            tau_slow=tau_slow,
            tau_fast=tau_fast,
            membrane_time_constant=membrane_time_constant,
            noise_rnn_sd=noise_rnn_sd,
            structure_mask=structure_mask,
            name='rnn')

        # Intialize output weights
        _w_out = initializers.gamma([n_hidden, n_output], connection_prob)
        inh_mask = np.ones_like(_w_out)
        inh_mask[-self.rnn.num_inh_units:, :] = 0.
        _w_out *= inh_mask
        self.W_out = tf.Variable(_w_out, dtype=tf.float32, name='W_out')
        self.b_out = tf.Variable(tf.zeros([1, n_output]), name='b_out')

    @tf.function
    def call(self, x, training=False):
        """
        This carries out an entire trial.
        x has shape (Tsteps, Batch, n_input).
        """
        tsteps = x.shape[0]
        batch_size = x.shape[1]
        n_hidden = self.rnn.n_hidden
        h = 0.1*tf.ones([batch_size, n_hidden])
        syn_x = tf.ones([batch_size, n_hidden])
        syn_u = tf.stack([tf.squeeze(tf.constant(self.rnn.U)) for \
            i in range(batch_size)])
        logits_list = []
        h_list = []
        syn_x_list = []
        syn_u_list = []
        for obs in tf.unstack(x):
            h, [h, syn_x, syn_u] = self.rnn(obs, [h, syn_x, syn_u])
            logits = h @ tf.nn.relu(self.W_out) + self.b_out
            logits_list.append(logits)
            h_list.append(h)
            syn_x_list.append(syn_x)
            syn_u_list.append(syn_u)
        return_dict = {}
        return_dict['logits'] = tf.stack(logits_list)
        return_dict['h'] = tf.stack(h_list)
        return_dict['syn_x'] = tf.stack(syn_x_list)
        return_dict['syn_u'] = tf.stack(syn_u_list)
        return return_dict

    @tf.function
    def train_step(self, opt, data, spike_cost_coef=1e-2, weight_cost_coef=0):
        [x, mask], y_true = data
        with tf.GradientTape() as tape:
            results = self(x, training=True)
            xe_loss = losses.cross_entropy(y_true, results['logits'], mask)
            if spike_cost_coef > 0:
                spike_cost = spike_cost_coef*losses.spike_cost(results['h'])
            else:
                spike_cost = 0
            if weight_cost_coef > 0:
                weight_cost = weight_cost_coef*losses.weight_cost(self.rnn.W_rnn)
            else:
                weight_cost = 0
            loss = xe_loss + spike_cost + weight_cost
        grads = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grads, self.trainable_variables))
        acc = metrics.accuracy(y_true, results['logits'], mask)
        metrics_dict = {
            'accuracy': acc,
            'loss': loss,
            'xe_loss': xe_loss,
            'spike_cost': spike_cost,
            'weight_cost': weight_cost
        }
        return metrics_dict

    def test_step(self, data):
        [x, mask], y_true = data
        results = self(x, training=False)
        acc = metrics.accuracy(y_true, results['logits'], mask=mask)
        return {'accuracy': acc}

    def record_step(self, data):
        [x, mask], y_true = data
        results = self(x, training=False)
        results['accuracy'] = metrics.accuracy(y_true, logits_seq, mask=mask)
        results['predictions'] = tf.math.argmax(logits_seq, axis=2) # (T, B)
        return results

    def load_npy_weights(self, w_in=None, w_rnn=None, b_rnn=None, w_out=None, b_out=None):
        """
        Loads .npy weights to model.
        """
        if w_in is not None:
            self.rnn.W_in.assign(w_in)
        if w_rnn is not None:
            self.rnn.W_rnn.assign(w_rnn)
        if b_rnn is not None:
            self.rnn.b_rnn.assign(b_rnn)
        if w_out is not None:
            self.W_out.assign(w_out)
        if b_out is not None:
            self.b_out.assign(b_out)

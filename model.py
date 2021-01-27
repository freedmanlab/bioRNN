import tensorflow as tf
import sonnet as snt
import numpy as np

class bioRNN(snt.RNNCore):

    def __init__(self, hidden_size, output_size, EI=True,
        excitatory_frac=0.8, balance_EI=True, connection_prob=1.,
        synapse_config='full', n_receptive_fields=2., dt=10, tau_slow=1500.,
        tau_fast=200., membrane_time_constant=100., noise_rnn_sd=0.5, name=None):

        super().__init__(name=name)

        # Copy params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.EI = EI
        self.balance_EI = balance_EI
        self.connection_prob = connection_prob
        self.synapse_config = synapse_config

        # Compute other params
        self.dt_sec = tf.constant(dt/1000)
        self.alpha_neuron = tf.constant(dt/membrane_time_constant)
        self.noise_rnn_sd = tf.math.sqrt(2*self.alpha_neuron)*noise_rnn_sd
        self.input_connection_prob = connection_prob/n_receptive_fields

        # EI stuff
        self.n_exc_units = round(excitatory_frac*hidden_size)
        self.n_inh_units = hidden_size - self.n_exc_units
        self.EI_list = [1. for i in range(self.n_exc_units)] + \
            [-1. for i in range(self.n_inh_units)]
        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))

        # STP stuff
        synapse_config_list = self._get_synapse_config_list(synapse_config)
        self.alpha_stf = np.ones([1, hidden_size], dtype=np.float32)
        self.alpha_std = np.ones([1, hidden_size], dtype=np.float32)
        self.U = np.ones([1, hidden_size], dtype=np.float32)
        self.dynamic_synapse = np.zeros([1, hidden_size], dtype=np.float32)
        for i in range(hidden_size):
            # If static, leave alone
            if synapse_config_list[i] == 'facilitating':
                self.alpha_stf[0,i] = dt/tau_slow
                self.alpha_std[0,i] = dt/tau_fast
                self.U[0,i] = 0.15
                self.dynamic_synapse[0,i] = 1.
            elif synapse_config_list[i] == 'depressing':
                self.alpha_stf[0,i] = dt/tau_fast
                self.alpha_std[0,i] = dt/tau_slow
                self.U[0,i] = 0.45
                self.dynamic_synapse[0,i] = 1.

    @snt.once
    def _initialize(self, input):
        # Get shape
        self.input_size = tf.shape(input)[1]

        # Initial values for variables
        w_in_init = self._gamma([self.input_size, self.hidden_size],
            self.input_connection_prob)
        w_rnn_init = self._gamma([self.hidden_size, self.hidden_size],
            self.connection_prob)
        w_out_init = self._gamma([self.hidden_size, self.output_size],
            self.connection_prob, alpha=0.1)
        if self.synapse_config == 'static':
            w_in_init /= 100
            w_rnn_init /= 100
            w_out_init /= 100
        else:
            w_in_init /= 10
            w_rnn_init /= 10
            w_out_init /= 10

        # Initialize variables
        self.w_in = tf.Variable(w_in_init, name='w_in')
        self.w_rnn = tf.Variable(w_rnn_init, name='w_rnn')
        self.b_rnn = tf.Variable(tf.zeros([1, self.hidden_size]), name='b_rnn')
        self.w_out = tf.Variable(w_out_init, name='w_out')

    def initial_state(self, batch_size):
        h_init = 0.1*tf.ones([batch_size, self.hidden_size])
        syn_x_init = tf.ones([batch_size, self.hidden_size])
        syn_u_init = tf.tile(self.U, [batch_size, 1])
        return [h_init, syn_x_init, syn_u_init]

    def __call__(self, x, state, h_mask=None, w_rnn_mask=None):
        """
        Does a single timestep.
        - x is input at single timestep, with shape [B, input_size]
        - h_mask has shape (1, hidden_size)
        - w_rnn_mask has shape (hidden_size, hidden_size)
        """
        self._initialize(x)
        h, syn_x, syn_u = state

        # Update STP state
        if self.synapse_config == 'static':
            h_post = h
        else:
            syn_x += self.dynamic_synapse*(self.alpha_std*(1-syn_x) - self.dt_sec*syn_u*syn_x*h)
            syn_u += self.dynamic_synapse*(self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h)
            syn_x = tf.math.minimum(1., tf.nn.relu(syn_x))
            syn_u = tf.math.minimum(1., tf.nn.relu(syn_u))
            h_post = syn_u * syn_x * h

        # If no masks, set to ones
        if h_mask is None:
            h_mask = tf.ones_like(h_post)
        if w_rnn_mask is None:
            w_rnn_mask = tf.ones_like(self.w_rnn) - tf.linalg.diag(tf.ones(self.hidden_size))

        # Compute updates
        leaky_part = (1-self.alpha_neuron)*h
        update = self.alpha_neuron*(x @ tf.nn.relu(self.w_in)
            + (h_post*h_mask) @ self.EI_matrix @ tf.nn.relu(self.w_rnn*w_rnn_mask)
            + self.b_rnn)
        noise = tf.random.normal(tf.shape(h), mean=0., stddev=self.noise_rnn_sd)
        h = tf.nn.relu(leaky_part + update + noise)
        output = h @ self.w_out
        return output, [h, syn_x, syn_u]

    def do_trial(self, inputs, h_mask=None, w_rnn_mask=None):
        """
        Does an entire trial.
        - inputs has shape (T, B, input_size)
        - h_mask has shape (1, hidden_size)
        - w_rnn_mask has shape (hidden_size, hidden_size)
        """
        # Initial state
        batch_size = tf.shape(inputs)[1]
        state = self.initial_state(batch_size)

        # Lists to record outputs and state
        y_list = []
        h_list = []
        syn_x_list = []
        syn_u_list = []

        # Do trial
        for x in tf.unstack(inputs):
            y, state = self(x, state, h_mask=h_mask, w_rnn_mask=w_rnn_mask)
            y_list.append(y)
            h_list.append(state[0])
            syn_x_list.append(state[1])
            syn_u_list.append(state[2])

        # Return results
        results = {
            'outputs': tf.stack(y_list),
            'h': tf.stack(h_list),
            'syn_x': tf.stack(syn_x_list),
            'syn_u': tf.stack(syn_u_list)
        }
        return results

    def _gamma(self, dims, connection_prob, alpha=0.2, beta=1.0):
        w = tf.random.gamma(dims, alpha=alpha, beta=beta)
        prune_mask = tf.cast((tf.random.uniform(dims) < connection_prob), tf.float32)
        return w * prune_mask

    def _get_synapse_config_list(self, synapse_config):
        d = {
            'full': ['facilitating' if i%2==0 else 'depressing' for i in
                range(self.hidden_size)],
            'fac': ['facilitating' for i in range(self.hidden_size)],
            'dep': ['depressing' for i in range(self.hidden_size)],
            'static': ['static' for i in range(self.hidden_size)],
            'exc_fac': ['facilitating' if self.EI_list[i]==1 else 'static'
                for i in range(self.hidden_size)],
            'exc_dep': ['depressing' if self.EI_list[i]==1 else 'static'
                for i in range(self.hidden_size)],
            'inh_fac': ['facilitating' if self.EI_list[i]==-1 else 'static'
                for i in range(self.hidden_size)],
            'inh_dep': ['depressing' if self.EI_list[i]==-1 else 'static'
                for i in range(self.hidden_size)],
            'exc_dep_inh_fac': ['depressing' if self.EI_list[i]==1 else
                'facilitating' for i in range(self.hidden_size)]
        }
        return d[synapse_config]

import tensorflow as tf

def cross_entropy(y_true, y_logits, mask=None):
    """
    - y_true, y_logits have shape (Tsteps, Batch, n_pol)
    - mask has shape (Tsteps, Batch)
    """
    if mask is None: mask = tf.ones([y_true.shape[0], y_true.shape[1], 1])
    return tf.reduce_mean(mask*tf.nn.softmax_cross_entropy_with_logits(
        y_true, y_logits, axis=-1))

def spike_cost(h, norm=2):
    return tf.reduce_mean(h**norm)

def weight_cost(w, norm=2):
    return tf.reduce_mean(tf.nn.relu(w)**norm)

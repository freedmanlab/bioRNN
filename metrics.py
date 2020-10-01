import tensorflow as tf

def accuracy(y_true, y_logits, mask=None):
    """
    - y_true, y_logits have shape (Tsteps, Batch, n_pol)
    - mask has shape (Tsteps, Batch)
    """
    if mask is None: mask = tf.ones([y_true.shape[0], y_true.shape[1]])
    mask_test = mask*tf.cast((y_true[:,:,0]==0), tf.float32) # (T, B)
    len_mask = tf.math.count_nonzero(mask_test[:,0])
    preds = tf.math.argmax(y_logits, axis=2) # (T, B)
    preds = tf.one_hot(preds, y_true.shape[2], axis=-1) # (T, B, n_pol)
    mask_test = tf.expand_dims(mask_test, axis=-1) # (T, B, 1)
    count_true = tf.math.count_nonzero(mask_test*preds*y_true, axis=-1) # (T, B)
    return tf.reduce_mean(tf.reduce_sum(count_true, axis=0)/len_mask)

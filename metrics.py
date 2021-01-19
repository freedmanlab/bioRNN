import tensorflow as tf

def decision_accuracy(y_true, y_logits):
    """ Both have shape (T, B, n_pol) """
    n_pol = tf.shape(y_true)[-1]
    fixation_idx = tf.cast(n_pol-1, tf.int64)
    y_pred_idx = tf.math.argmax(y_logits, axis=-1) # (T, B)
    y_true_idx = tf.math.argmax(y_true, axis=-1) # (T, B)
    decision_mask = tf.cast(y_true_idx<fixation_idx, tf.float32) # (T, B)
    decision_len = tf.math.reduce_sum(decision_mask, axis=0) # (B,)
    bools = tf.cast(y_pred_idx==y_true_idx, tf.float32) # (T, B)
    bools *= decision_mask # (T,B)
    acc = tf.math.reduce_sum(bools) / tf.math.reduce_sum(
        tf.cast(decision_mask > 0., tf.float32))
    return acc

def fixation_accuracy(y_true, y_logits):
    """ Both have shape (T, B, n_pol) """
    n_pol = tf.shape(y_true)[-1]
    fixation_idx = tf.cast(n_pol-1, tf.int64)
    y_pred_idx = tf.math.argmax(y_logits, axis=-1) # (T, B)
    y_true_idx = tf.math.argmax(y_true, axis=-1) # (T, B)
    mask = tf.cast(y_true_idx==fixation_idx, tf.float32) # (T, B)
    fixation_len = tf.math.reduce_sum(mask, axis=0) # (B,)
    bools = tf.cast(y_pred_idx==y_true_idx, tf.float32) # (T, B)
    bools *= mask # (T,B)
    acc = tf.math.reduce_sum(bools) / tf.math.reduce_sum(
        tf.cast(mask > 0., tf.float32))
    return acc

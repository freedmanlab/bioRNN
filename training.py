import tensorflow as tf
from metrics import decision_accuracy, fixation_accuracy

@tf.function
def supervised_train_step(rnn, opt, inputs, labels, vars=None,
    train_mask=None, h_mask=None, w_rnn_mask=None,
    spike_cost_coef=1e-2, weight_cost_coef=0):
    """
    Does a supervised train step on an entire batch of trials.
    - inputs has shape (T, B, input_size)
    - labels has shape (T, B, output_size)
    - if vars is None, defaults to all trainable variables
    - train_mask has shape (T, B)
    """
    # Do trial
    with tf.GradientTape() as tape:
        results = rnn.do_trial(inputs, h_mask=h_mask,
            w_rnn_mask=w_rnn_mask)

        # Compute softmax cross entropy
        logits = results['outputs']
        logits -= tf.expand_dims(tf.math.reduce_max(logits, axis=-1), axis=-1) # prevent overflow
        Z = tf.math.reduce_sum(tf.math.exp(logits), axis=-1) # (T, B)
        log_p = tf.math.reduce_sum(logits*labels, axis=-1) # (T, B)
        cross_entropy_loss = -tf.math.reduce_mean(train_mask*(log_p-tf.math.log(Z)))

        # tf.print(tf.math.reduce_max(results['h']))

        # Compute spike and weight costs
        spike_cost = spike_cost_coef*tf.math.reduce_mean(
            tf.math.square(results['h']))
        weight_cost = weight_cost_coef*tf.math.reduce_mean(
            tf.math.square(tf.nn.relu(rnn.w_rnn)))

        loss = cross_entropy_loss + spike_cost + weight_cost

    # Compute and apply gradients
    if vars is None:
        vars = rnn.trainable_variables
    grads = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grads, vars))

    # Return results
    argmax_acc = tf.keras.metrics.categorical_accuracy(labels, logits)
    argmax_acc = tf.math.reduce_mean(argmax_acc)
    decision_acc = decision_accuracy(labels, logits)
    fixation_acc = fixation_accuracy(labels, logits)
    metrics = {
        'argmax_accuracy': argmax_acc,
        'decision_accuracy': decision_acc,
        'fixation_accuracy': fixation_acc,
        'loss': loss,
        'cross_entropy_loss': cross_entropy_loss,
        'spike_cost': spike_cost,
        'weight_cost': weight_cost
    }
    return metrics

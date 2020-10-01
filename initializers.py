import numpy as np

def gamma(dims, connection_prob, shape_param=0.1, scale_param=1.0):
    """
    Sample weights from Gamma distribution, then prune according to
    connection_prob.

    - dims: [num_row, num_col] for weight matrix
    - connection_prob: scalar in [0,1]
    - shape_param, scale_param are parameters for the Gamma distribution
    """
    w_ = np.random.gamma(shape_param, scale=scale_param, size=dims)
    prune_mask = (np.random.uniform(size=dims) < connection_prob)
    return w_ * prune_mask

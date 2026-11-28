"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np
def numerical_gradient(f, x, eps=1e-5):
    x = np.array(x)
    gradients = np.zeros_like(x,dtype=float)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_pos = f(x)
        x[idx] = orig - eps
        f_neg = f(x)
        grad = (f_pos - f_neg)/(2*eps)
        gradients[idx] = grad
        x[idx] = orig
    return gradients

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad,dtype=float)
    numeric_grad = np.asarray(numeric_grad,dtype=float)
    abs_diff = np.abs(analytic_grad - numeric_grad)
    deno = np.maximum(np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)),tol)
    rel_error = abs_diff/deno
    return np.max(rel_error)

# Step 3 - make_dense
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}

    def forward(x):
        y = x @ params['W'] + params['b']
        return y, x

    def backward(dout, cache):
        x = cache
        dx = dout @ params['W'].T
        dW = x.T @ dout
        db = np.sum(dout, axis=0)
        return dx, {'W': dW, 'b': db}

    return {'params': params, 'forward': forward, 'backward': backward}

# Step 4 - make_activation
import numpy as np

def make_activation(kind='relu'):
    if kind != 'relu':
        raise ValueError(f"Unsupported activation kind: {kind}")

    def forward(x):
        y = np.maximum(0, x)
        return y, x

    def backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx, {}

    return {'params': {}, 'forward': forward, 'backward': backward}

# Step 5 - initialize_weights
import numpy as np

def initialize_weights(in_dim, out_dim, scheme='he'):
    if scheme == 'he':
        scale = np.sqrt(2.0 / in_dim)
    elif scheme == 'xavier':
        scale = np.sqrt(2.0 / (in_dim + out_dim))
    else:
        raise ValueError(f"Unsupported initialization scheme: {scheme}")

    W = np.random.randn(in_dim, out_dim) * scale
    b = np.zeros(out_dim)
    return W, b

# Step 6 - make_loss
import numpy as np

def make_loss(kind='cross_entropy'):
    if kind != 'cross_entropy':
        raise ValueError(f"Unsupported loss kind: {kind}")

    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        N = logits.shape[0]

        # Shift logits by row max for numerical stability (prevents exp overflow)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Mean negative log-likelihood of the true class
        correct_log_probs = -np.log(np.clip(probs[np.arange(N), labels], 1e-15, 1.0))
        loss = float(np.mean(correct_log_probs))

        # Analytic gradient: (probs - one_hot(labels)) / N
        d_logits = probs.copy()
        d_logits[np.arange(N), labels] -= 1.0
        d_logits /= N

        return loss, d_logits

    return loss_fn

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement


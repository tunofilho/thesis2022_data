import tensorflow as tf
# from gctf import optimizers as gc_opt
# from gctf import centralized_gradients_for_optimizer as gc_opt
from tensorflow.keras.optimizers import Adam, SGD

# credits : keras.io site

class GC_SGD(SGD):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

class GC_ADAM(Adam):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

def opt_model(fn, hiper):
    if hiper.optimizer == 'sgd':
        if hiper.gc:
            # opt = gc_opt.sgd(learning_rate=fn, momentum=0.9)
            opt = GC_SGD(learning_rate=fn, momentum=0.9)

        else:
            opt = tf.keras.optimizers.SGD(learning_rate=fn, momentum=0.9)

        return opt
    else:
        if hiper.gc:
            # opt = gc_opt.adam(learning_rate=fn)
            opt = GC_ADAM(learning_rate=fn)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=fn)

        return opt
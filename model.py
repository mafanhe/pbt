# import numpy as np


def dnn_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.nn import relu, softmax
    model = Sequential([

        Flatten(input_shape=(28, 28)),
        Dense(512, activation=relu),
        Dropout(0.2),
        Dense(10, activation=softmax)
    ])
    return model


def get_optimizer():
    import numpy as np
    from tensorflow.keras.optimizers import SGD
    from utils import choose
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = SGD
    lr = choose(np.logspace(-5, 0, base=10))
    momentum = choose(np.linspace(0.1, .9999))
    return optimizer_class(lr=lr, momentum=momentum)

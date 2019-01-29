from utils import choose


def dnn_model():
    global tf
    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


def get_optimizer():
    global tf
    from numpy import logspace, linspace
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = tf.keras.optimizers.SGD
    lr = choose(logspace(-5, 0, base=10))
    momentum = choose(linspace(0.1, .9999))
    return optimizer_class(lr=lr, momentum=momentum)

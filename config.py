HYPERPARAM_NAMES = ["lr", "momentum"]  # This is unfortunate.
EPOCHS = 10
BATCH_SIZE = 64
POPULATION_SIZE = 5  # Number of models in a population
EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs
USE_SQLITE = True  # If False, you'll need to set up a local Postgres server
# tf.enable_eager_execution()
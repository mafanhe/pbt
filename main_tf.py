from utils import (choose, split_trn_val, update_task, get_max_of_db_column,
                   get_a_task, get_task, ExploitationNeeded,
                   LossIsNaN, get_task_ids_and_scores, PopulationFinished,
                   get_col_from_populations, RemainingTasksTaken,
                   print_with_time, ExploitationOcurring,
                   create_new_population)
import argparse
import os
import time
import pathlib
from psycopg2.extensions import TransactionRollbackError
import numpy as np
import pickle
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

HYPERPARAM_NAMES = ["lr", "momentum"]  # This is unfortunate.
EPOCHS = 10
BATCH_SIZE = 64
POPULATION_SIZE = 5  # Number of models in a population
EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs
USE_SQLITE = True  # If False, you'll need to set up a local Postgres server


# tf.enable_eager_execution()


def data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def dnn_model():
    global tf
    # from tensorflow.keras.layers import Flatten,Dense, Dropout
    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


def get_optimizer():
    global tf
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = tf.keras.optimizers.SGD
    lr = choose(np.logspace(-5, 0, base=10))
    momentum = choose(np.linspace(0.1, .9999))
    return optimizer_class(lr=lr, momentum=momentum)


def save_obj(obj, path):
    out_put = open(path, "wb")
    pickle.dump(obj, out_put)


def load_obj(path):
    read_file = open(path, "rb")
    return pickle.load(read_file)


class Trainer:

    def __init__(self, model=None, optimizer=None, x_train=None, y_train=None, x_test=None, y_test=None,
                 epochs=1, batch_size=None, valid_size=0.2, task_id=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        if x_train is not None:
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.epochs = epochs
            num_examples = len(self.y_train)
            self.trn_indices, self.val_indices = \
                split_trn_val(num_examples, valid_size)
            # Sometimes we only use a Trainer to load and save checkpoints.
            #   When that's the case, we don't need the following.
            self.batch_size = batch_size
        self.task_id = task_id

    def save_checkpoint(self, checkpoint_path):
        # checkpoint = dict(model_state_dict=self.model.get_weights(),
        #                   optim_state_dict=self.optimizer.get_weights())
        # save_obj(self.optimizer.get_params(),checkpoint_path+".params")
        # torch.save(checkpoint, checkpoint_path)
        self.model.save(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        global tf

        # checkpoint = torch.load(checkpoint_path)
        # self.model.set_weights(checkpoint['model_state_dict'])
        # params = load_obj(checkpoint_path+".params")
        # self.optimizer.set_params_weights(params,checkpoint['optim_state_dict'])
        self.model = tf.keras.models.load_model(checkpoint_path)

    def train(self, second_half, seed_for_shuffling):
        global tf

        print('Train(task % d) ' % self.task_id)
        # TODO shuffle train data
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)]
        self.model.compile(optimizer=self.optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['acc'])

        self.model.fit(self.x_train[self.trn_indices],
                       self.y_train[self.trn_indices],
                       epochs=self.epochs,
                       callbacks=callbacks,
                       validation_data=(self.x_train[self.val_indices],
                                        self.y_train[self.val_indices]),
                       verbose=2,  # Logs once per epoch.
                       batch_size=self.batch_size)

    def eval(self, interval_id):
        """Evaluate model on the provided validation or test set."""
        print('Eval (interval %d)' % interval_id)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
        print("accuracy:%f" % acc)
        return acc

    def exploit_and_explore(self, better_trainer, hyperparam_names,
                            perturb_factors=[1.2, 0.8]):
        """Copy parameters from the better model and the hyperparameters
           and running averages from the corresponding optimizer."""
        # Copy model parameters
        better_model = better_trainer.model
        better_state_dict = better_model.get_weights()
        self.model.set_weights(better_state_dict)
        # Copy optimizer state (includes hyperparameters and running averages)
        better_optimizer = better_trainer.model.optimizer
        # Assumption: Same LR and momentum for each param group
        # Perturb hyperparameters

        param_group = better_optimizer.get_config()
        for hyperparam_name in hyperparam_names:
            perturb = np.random.choice(perturb_factors)
            param_group[hyperparam_name] *= perturb
        self.optimizer = self.optimizer.from_config(param_group)


def init():
    global tf
    global sess
    import tensorflow as tf
    import tensorflow.keras.backend as KTF
    tf.logging.set_verbosity(tf.logging.DEBUG)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


def exploit_and_explore(connect_str_or_path, population_id):
    intervals_trained_col = get_col_from_populations(
        connect_str_or_path, USE_SQLITE,
        population_id, "intervals_trained")
    intervals_trained_col = np.array(intervals_trained_col)
    if not np.all(
            intervals_trained_col == intervals_trained_col[0]):
        msg = """The exploiter seems to be exploiting before all
                 the models have finished training.
                 Check for bad race conditions with respect
                 to the database."""
        raise Exception(msg)
    # Sorted by scores, desc
    task_ids, scores = get_task_ids_and_scores(connect_str_or_path,
                                               USE_SQLITE,
                                               population_id)
    print_with_time("Exploiting interval %s. Best score: %.4f" %
                    (intervals_trained_col[0] - 1, max(scores)))
    seed_for_shuffling = np.random.randint(10 ** 5)
    fraction = 0.20
    cutoff = int(np.ceil(fraction * len(task_ids)))
    top_ids = task_ids[:cutoff]
    bottom_ids = task_ids[len(task_ids) - cutoff:]
    nonbottom_ids = task_ids[:len(task_ids) - cutoff]
    for bottom_id in bottom_ids:
        top_id = np.random.choice(top_ids)
        model = dnn_model()
        optimizer = get_optimizer()
        top_trainer = Trainer(model=model,
                              optimizer=optimizer)
        top_checkpoint_path = (checkpoint_str %
                               (population_id, top_id))
        top_trainer.load_checkpoint(top_checkpoint_path)
        model = dnn_model()
        optimizer = get_optimizer()
        bot_trainer = Trainer(model=model,
                              optimizer=optimizer)
        bot_checkpoint_path = (checkpoint_str %
                               (population_id, bottom_id))

        bot_trainer.load_checkpoint(bot_checkpoint_path)
        bot_trainer.exploit_and_explore(top_trainer,
                                        HYPERPARAM_NAMES)
        bot_trainer.save_checkpoint(bot_checkpoint_path)
        key_value_pairs = dict(
            ready_for_exploitation=ready_for_exploitation_False,
            score=None,
            seed_for_shuffling=seed_for_shuffling)
        update_task(connect_str_or_path, USE_SQLITE,
                    population_id, bottom_id, key_value_pairs)
    for nonbottom_id in nonbottom_ids:
        key_value_pairs = dict(
            ready_for_exploitation=ready_for_exploitation_False,
            seed_for_shuffling=seed_for_shuffling)
        update_task(connect_str_or_path, USE_SQLITE,
                    population_id, nonbottom_id, key_value_pairs)
    del trainer.model
    del trainer
    tf.keras.backend.clear_session()


def tran(x_train, y_train, x_test, y_test, epochs, batch_size, task_id, population_id,
         ready_for_exploitation_False,
         ready_for_exploitation_True,
         active_False,
         active_True,
         connect_str_or_path,
         intervals_trained, seed_for_shuffling):
    # Train
    print(os.getpid())
    optimizer = get_optimizer()
    model = dnn_model()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs,
        batch_size=batch_size,
        task_id=task_id)

    checkpoint_path = (checkpoint_str %
                       (population_id, task_id))

    if os.path.isfile(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    interval_is_odd = intervals_trained % 2 == 1
    score = None
    try:
        try:
            trainer.train(interval_is_odd, seed_for_shuffling)
            time.sleep(1)
        except LossIsNaN:
            print_with_time("Setting score to -1.")
            score = -1
        if score != -1:
            score = float(trainer.eval(intervals_trained))
            trainer.save_checkpoint(checkpoint_path)
        key_value_pairs = dict(
            intervals_trained=intervals_trained + 1,
            ready_for_exploitation=ready_for_exploitation_True,
            active=active_False,
            score=score)
        update_task(connect_str_or_path, USE_SQLITE,
                    population_id, task_id, key_value_pairs)
        sess.close()
        del trainer.model
        del trainer
        tf.keras.backend.clear_session()
    except KeyboardInterrupt:
        # Don't save work.
        key_value_pairs = dict(active=active_False)
        update_task(connect_str_or_path, USE_SQLITE,
                    population_id, task_id, key_value_pairs)
        sess.close()
        del trainer.model
        del trainer
        tf.keras.backend.clear_session()
        # break


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.WARN)

    print("mian:",os.getpid())

    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-p", "--population_id", type=int, default=None,
                        help="Resumes work on the population with the given ID. Use -1 to select the most recently created population. Without this flag, a new population will be created.")
    parser.add_argument("-e", "--exploiter", action="store_true",
                        help="Set this process as the exploiter. It will be responsible for running the exploit step over the entire population at the end of each interval.")
    args = parser.parse_args()

    population_id = args.population_id
    exploiter = args.exploiter
    (x_train, y_train), (x_test, y_test) = data()
    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_str = "checkpoints/pop-%03d_task-%03d.h5"
    interval_limit = int(np.ceil(EPOCHS / EXPLOIT_INTERVAL))
    table_name = "populations"
    if USE_SQLITE:
        sqlite_path = "database.sqlite3"
        connect_str_or_path = sqlite_path
        ready_for_exploitation_False = 0
        ready_for_exploitation_True = 1
        active_False = 0
        active_True = 1
    else:  # Postgres
        db_env_var_names = ['PGDATABASE', 'PGUSER', 'PGPORT', 'PGHOST']
        db_params = [os.environ[var_name] for var_name in db_env_var_names]
        db_connect_str = "dbname={} user={} port={} host={}".format(*db_params)
        connect_str_or_path = db_connect_str
        ready_for_exploitation_False = False
        ready_for_exploitation_True = True
        active_False = False
        active_True = True
    if population_id is None:
        population_id = create_new_population(connect_str_or_path, USE_SQLITE,
                                              POPULATION_SIZE)
        msg = "Population added to populations table. Population ID: %s"
        print_with_time(msg % population_id)
    elif population_id == -1:
        population_id = get_max_of_db_column(connect_str_or_path, USE_SQLITE,
                                             table_name, "population_id")
    # Train each available task for an interval
    task_wait_count = 0
    exploitation_wait_count = 0
    start_time = int(time.time())
    # global exploitation_wait_count
    # global task_wait_count
    while True:

        try:
            # Find a task that's incomplete and inactive, and set it to active
            pool = multiprocessing.Pool(processes=POPULATION_SIZE, initializer=init)  #

            try:
                tasks = get_task(connect_str_or_path, USE_SQLITE, population_id,
                                 interval_limit, POPULATION_SIZE)
                # task_id, intervals_trained, seed_for_shuffling = task
            except RemainingTasksTaken:
                if task_wait_count == 0:
                    print_with_time("Waiting for a task to be available.")
                time.sleep(10)
                task_wait_count += 1
                continue
            except PopulationFinished:
                task_ids, scores = get_task_ids_and_scores(connect_str_or_path,
                                                           USE_SQLITE,
                                                           population_id)
                print("Population finished. Best score: %.2f" % scores[0])
                checkpoint_path = (checkpoint_str % (population_id, task_ids[0]))
                pre, suf = checkpoint_path.split('.')
                weights_path = pre + "_weights." + suf
                print("Best weights saved to: %s" % weights_path)
                break
            except (ExploitationNeeded, ExploitationOcurring):
                if exploiter:
                    pool.apply_async(exploit_and_explore, (connect_str_or_path, population_id))
                    pool.close()
                    pool.join()
                    time.sleep(1)
                    continue
                else:
                    print_with_time("Waiting for exploiter to finish.")
                    time.sleep(10)
                    exploitation_wait_count += 1
                    if exploitation_wait_count > 11:
                        print_with_time(
                            "Exploiter is taking too long. Ending process.")
                        quit()
                    continue
            except TransactionRollbackError:
                print_with_time("Deadlock?")
                time.sleep(1)
                continue

            # multiprocessing train  
            for task_id, intervals_trained, seed_for_shuffling in tasks:
                pool.apply_async(tran, (
                x_train, y_train, x_test, y_test, 1, BATCH_SIZE, task_id, population_id, ready_for_exploitation_False,
                ready_for_exploitation_True, active_False, active_True, connect_str_or_path, intervals_trained,
                seed_for_shuffling))
            pool.close()
            pool.join()
            # time.sleep(10)

        except KeyboardInterrupt:
            pool.terminate()
            break

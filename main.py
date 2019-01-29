import numpy as np
import os
import argparse
import time
import pathlib
import multiprocessing
from psycopg2.extensions import TransactionRollbackError
from utils import (update_task, get_max_of_db_column,
                   get_task, ExploitationNeeded,
                   LossIsNaN, get_task_ids_and_scores, PopulationFinished,
                   get_col_from_populations, RemainingTasksTaken,
                   print_with_time, ExploitationOcurring,
                   create_new_population)
from config import *
from model import dnn_model
from model import get_optimizer
from trainer import Trainer
from datasets import data

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}


def init():
    global tf
    global sess
    import tensorflow as tf
    import tensorflow.keras.backend as KTF

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


def train(x_train, y_train, x_test, y_test, epochs, batch_size, task_id, population_id,
          ready_for_exploitation_False,
          ready_for_exploitation_True,
          active_False,
          active_True,
          connect_str_or_path,
          intervals_trained, seed_for_shuffling):
    # Train

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

    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-p", "--population_id", type=int, default=None,
                        help="Resumes work on the population with the given ID. Use -1 to select the most recently "
                             "created population. Without this flag, a new population will be created.")
    parser.add_argument("-e", "--exploiter", action="store_true",
                        help="Set this process as the exploiter. It will be responsible for running the exploit step "
                             "over the entire population at the end of each interval.")
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
            # init process pool
            pool = multiprocessing.Pool(processes=POPULATION_SIZE, initializer=init)  #
            try:
                # Find a task that's incomplete and inactive, and set it to active
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
                pool.apply_async(train, (
                    x_train, y_train, x_test, y_test, 1, BATCH_SIZE, task_id, population_id,
                    ready_for_exploitation_False,
                    ready_for_exploitation_True, active_False, active_True, connect_str_or_path, intervals_trained,
                    seed_for_shuffling))
            pool.close()
            pool.join()
            # time.sleep(10)

        except KeyboardInterrupt:
            pool.terminate()
            break

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


class Trainer:

    def __init__(self, model=None, optimizer=None, x_train=None, y_train=None, x_test=None, y_test=None,
                 epochs=1, batch_size=None, valid_size=0.2, task_id=None):
        """Note: Trainer objects don't know about the database."""
        from utils import split_trn_val
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
        self.model.save(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.model = load_model(checkpoint_path)

    def train(self, second_half, seed_for_shuffling):

        print('Train(task % d) ' % self.task_id)
        # TODO shuffle train data
        callbacks = [EarlyStopping(
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
        # TODO
        param_group = better_optimizer.get_config()
        for hyperparam_name in hyperparam_names:
            perturb = np.random.choice(perturb_factors)
            param_group[hyperparam_name] *= perturb
        self.optimizer = self.optimizer.from_config(param_group)

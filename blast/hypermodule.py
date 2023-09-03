import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import Subset
from tqdm import tqdm

from .components import Trainer
from .partials import optim, scheds


class HyperModule:
    def __init__(self, model, criterion, optimizer, scheduler=None, hyperparams=None):
        self._optimizer, self._scheduler, self._hyperparams = None, None, None
        self.model, self.criterion = model, criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hyperparams = hyperparams

        # self.history = History()
        # self.train_loss, self.valid_loss, self.valid_acc = [], [], []
        # self.epoch_trained = 0
        # self.test_acc = None
        # self.load_path = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            self._optimizer = optimizer
        else:
            optim_gen = optim(optimizer)
            self._optimizer = optim_gen(params=self.model.parameters())
        if self._scheduler is not None:
            self.scheduler = sched(self._scheduler)

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self._scheduler = scheduler
        elif scheduler is not None:
            sched_gen = sched(scheduler)
            self._scheduler = sched_gen(optimizer=self._optimizer)

    @property
    def hyperparams(self):
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, hyperparams):
        if hyperparams:
            self._hyperparams = hyperparams
            self.optimizer = optim(self._optimizer, **hyperparams)
            if self._scheduler is not None:
                self.scheduler = sched(self._scheduler, **hyperparams)

    def train(self, train_dataloader, valid_dataloader, num_epochs, verbose):
        Trainer(self).fit(train_dataloader, valid_dataloader, num_epochs, verbose)

    def test(self, test_dataloader):
        raise NotImplementedError

    def save(
        self, checkpoint_dir, model=None, optimizer=None, scheduler=None, history=None, verbose=True
    ):
        if model:
            model_path = os.path.join(checkpoint_dir, model)
            torch.save(self.model.state_dict(), model_path)
        if optimizer:
            optimizer_path = os.path.join(checkpoint_dir, optimizer)
            torch.save(self._optimizer.state_dict(), optimizer_path)
        if self._scheduler and scheduler:
            scheduler_path = os.path.join(checkpoint_dir, scheduler)
            torch.save(self._scheduler.state_dict(), scheduler_path)
        if history:
            history_path = os.path.join(checkpoint_dir, history)
            with open(history_path, "w") as fp:
                json.dump(self.history, fp)
        if verbose:
            print("State dict saved.")

    def load(self, path, verbose=True):
        device = torch.device("cuda")
        if path == "last":
            path = self.load_path
        if path == "best":
            path = self.load_path + ".best"
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict["model"])
        self.model.to(device)
        self.model.eval()

        self._optimizer.load_state_dict(state_dict["optimizer"])
        if self._scheduler is not None:
            self._scheduler.load_state_dict(state_dict["scheduler"])
        else:
            self._scheduler = None
        self.test_acc = state_dict["test_acc"]

        n_train_loss = len(state_dict["train_loss"])
        n_valid_acc = len(state_dict["valid_acc"])
        epoch_trained = state_dict["epoch_trained"]
        n = min(epoch_trained, n_train_loss, n_valid_acc)

        self.epoch_trained = n
        self.train_loss = state_dict["train_loss"][:n]
        self.valid_acc = state_dict["valid_acc"][:n]
        if verbose:
            print("State dict sucessfully loaded.")

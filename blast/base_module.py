import os
from pathlib import Path

import torch
from torch.nn.functional import softmax

from .history import History
from .partials import optim, sched


class BaseModule:
    """The BaseModule contains the network architecture, loss, optimizer and scheduler.
    It also deals with the IO of these instances.

    TODO: load & save without knowing architectire in advance.
    (See. https://davidstutz.de/loading-and-saving-pytorch-models-without-knowing-the-architecture/)

    """

    def __init__(self, model, optimizer=None, scheduler=None, hyperparams=None):
        self._optimizer, self._scheduler, self._hyperparams = None, None, None
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hyperparams = hyperparams

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

    def save(self, model=None, optimizer=None, scheduler=None, verbose=True):
        if model:
            torch.save(self.model.state_dict(), model)
            if verbose:
                print("Model's state_dict saved:", model)
        if self._optimizer and optimizer:
            torch.save(self._optimizer.state_dict(), optimizer)
            if verbose:
                print("Optimizer's state_dict saved:", optimizer)
        if self._scheduler and scheduler:
            torch.save(self._scheduler.state_dict(), scheduler)
            if verbose:
                print("Scheduler's state_dict saved:", scheduler)

    def save_module(
        self,
        checkpoint_dir,
        model="model.ckpt",
        optimizer="optimizer.ckpt",
        scheduler="scheduler.ckpt",
        verbose=True,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / Path(model)
        optimizer_path = checkpoint_dir / Path(optimizer)
        scheduler_path = checkpoint_dir / Path(scheduler)
        self.save(model_path, optimizer_path, scheduler_path, verbose)

    def load(self, model=None, optimizer=None, scheduler=None, verbose=True):
        if model:
            self.model.load_state_dict(torch.load(model))
            self.model.to("cuda").eval()
            if verbose:
                print("Model is loaded from:", model)
        if self._optimizer and optimizer:
            self._optimizer.load_state_dict(torch.load(optimizer))
            if verbose:
                print("Optimizer is loaded from:", optimizer)
        if self._scheduler and scheduler:
            self._scheduler.load_state_dict(torch.load(scheduler))
            if verbose:
                print("Scheduler is loaded from:", scheduler)

    def load_module(
        self,
        checkpoint_dir,
        model="model.ckpt",
        optimizer="optimizer.ckpt",
        scheduler="scheduler.ckpt",
        verbose=True,
    ):
        model_path = str(Path(checkpoint_dir) / Path(model))
        optimizer_path = str(Path(checkpoint_dir) / Path(optimizer))
        scheduler_path = str(Path(checkpoint_dir) / Path(scheduler))
        self.load(model_path, optimizer_path, scheduler_path, verbose)

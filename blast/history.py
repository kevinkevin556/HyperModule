import json

import numpy as np


class EpochHistory:
    def __init__(self, loss, metrics):
        self.loss = {loss.__class__.__name__: []}
        self.metrics = {m: [] for m in metrics.keys()}

    def update(self, loss=None, metrics=None):
        if loss:
            for name, value in loss.items():
                self.loss[name].append(value)
        if metrics:
            for name, value in metrics.items():
                self.metrics[name].append(value)

    def __repr__(self) -> str:
        avg_train_loss = {key: np.mean(value) for key, value in self.loss.items()}
        avg_valid_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        out = "Train: "
        for key, value in avg_train_loss.items():
            out += f"{key} = {value:.3f} "
        out += "\tValid: "
        for key, value in avg_valid_metrics.items():
            out += f"{key} = {value:.3f} "
        return out


class History:
    def __init__(self, model, checkpoint_dir):
        self.model_name = model if isinstance(model, str) else model.__class__.__name__
        self.checkpoint_dir = checkpoint_dir
        self.training = []

    def new_training(self, loss, metrics, optimizer, scheduler):
        self.training.append(
            {
                "loss": {loss.__class__.__name__: []},
                "metrics": {m: [] for m in metrics.keys()},
                "optimizer": str(optimizer.__class__),
                "scheduler": str(scheduler.__class__) if scheduler else None,
                "n_epochs": 0,
            }
        )

    def update(self, loss=None, metrics=None, **kwargs):
        if loss:
            for name, value in loss.items():
                self.training[-1]["loss"][name].append(value)
        if metrics:
            for name, value in metrics.items():
                self.training[-1]["metrics"][name].append(value)
        if kwargs:
            self.training[-1].update(**kwargs)

    def to_dict(self):
        return {
            "model": self.model_name,
            "checkpoint_dir": self.checkpoint_dir,
            "training": self.training,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def load(self, path):
        with open(path, "r") as f:
            history_dict = json.load(f)
            self.model_name = history_dict["model"]
            self.checkpoint_dir = history_dict["checkpoint_dir"]
            self.training = history_dict["training"]

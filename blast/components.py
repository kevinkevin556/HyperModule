from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm.auto import tqdm

from .base_module import BaseModule
from .conditions import if_valid_loss_improved
from .history import EpochHistory, History


class Predictor(BaseModule):
    """Predictor class provides methods for prediction and pre-/post-processing.

    TODO: Add Preprocessing method
    TODO: Add Postporcessing method
    """

    def __init__(self, model, optimizer=None, scheduler=None, hyperparams=None):
        super().__init__(model, optimizer, scheduler, hyperparams)

    def predict(self, images=None, dataloader=None):
        self.model.to("cuda").eval()
        if images:
            images = images.to("cuda")
            return self.predict_image(images)
        if dataloader:
            return self.predict_dataloader(dataloader)

    def predict_image(self, images):
        self.model.to("cuda").eval()
        images = images.to("cuda")
        logits = self.model(images)
        probs = softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        return pred_labels

    def predict_dataloader(self, dataloader, target=False, logit=False):
        self.model.to("cuda").eval()
        pred_list, target_list = [], []
        with torch.no_grad():  # Turn off gradient computation to save memory
            for images, targets in dataloader:
                images, targets = images.to("cuda"), targets.to("cuda")
                predictions = self.model(images) if logit else self.predict_image(images)
                pred_list.append(predictions)
                if target:
                    target_list.append(targets)
        output = (torch.cat(pred_list), torch.cat(target_list)) if target else torch.cat(pred_list)
        return output


class Evaluator(Predictor):
    def __init__(
        self,
        *args,
        metrics={"Accuracy": lambda logits, y: (logits.argmax(1) == y).float().mean()},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def validate(self, dataloader):
        self.model.to("cuda").eval()
        logits, targets = self.predict_dataloader(dataloader, target=True, logit=True)
        output = {}
        for metric, metric_func in self.metrics.items():
            value = metric_func(logits, targets)
            output[metric] = value.item() if torch.is_tensor(value) else value
        return output


class Trainer(Evaluator):
    """
    TODO: ⭐ Earlystopping
    """

    def __init__(self, *args, loss, checkpoint_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.checkpoint_dir = checkpoint_dir
        self.history = History(self.model, self.checkpoint_dir)

        # Add training loss to metrics as validation loss
        self.metrics = {loss.__class__.__name__: loss} | self.metrics

    def fit(self, train_dataloader, valid_dataloader=None, max_epochs=1, verbose=True, **kwargs):
        self.model.to("cuda")
        self.history.new_training(self.loss, self.metrics, self.optimizer, self.scheduler)
        self.history.update(max_epochs=max_epochs)

        for epoch in range(max_epochs):
            self.epoch_history = EpochHistory(loss=self.loss, metrics=self.metrics)
            self.model.train()
            train_progress = tqdm(train_dataloader, position=0, leave=verbose)
            for images, targets in train_progress:
                images, targets = images.to("cuda"), targets.to("cuda")
                self.update(images, targets)
                self.update_progressbar(train_progress, max_epochs)
            self.perform_validation(valid_dataloader, verbose)
            self.update_scheduler()
            self.update_history()
            # Save last model
            self.checkpoint(
                model={"last_model": "last_model.ckpt"},
                optimizer="optimizer_state.pt",
                scheduler="scheduler_state.pt",
                verbose=False,
            )
            # Save model when validation loss is improved
            self.checkpoint(
                model={"best_model": "best_model.ckpt"}, condition=if_valid_loss_improved
            )

            # TODO: Make checkpointing more flexible??

        self.model.eval()
        del self.epoch_history

    def update(self, images, targets, **kwargs):
        ## back-propagate
        preds = self.model(images)
        loss = self.loss(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ## record loss
        self.epoch_history.update(loss={self.loss.__class__.__name__: loss.detach().item()})

    def update_progressbar(self, progress, max_epochs):
        ## Update progress bar description
        n_epochs = sum([t["n_epochs"] for t in self.history.training])
        total_epochs = sum([t["n_epochs"] for t in self.history.training[:-1]]) + max_epochs
        progress.set_description(f"Epoch [{n_epochs+1}/{total_epochs}]")

        ## Update progress bar postfix
        loss = self.epoch_history.loss[self.loss.__class__.__name__][-1]
        lr = getattr(self.scheduler, "_last_lr", [self.optimizer.defaults["lr"]])[0]
        postfix = {"loss": loss, "lr": lr} if self.scheduler else {"loss": loss}
        progress.set_postfix(postfix)

    def update_scheduler(self):
        if self.scheduler is None:
            pass
        elif "metrics" in self.scheduler.step.__code__.co_varnames:
            metrics = np.mean(self.epoch_history.metrics[self.loss.__class__.__name__][-1])
            self.scheduler.step(metrics=metrics)
        else:
            self.scheduler.step()

    def update_history(self):
        n_epochs = self.history.training[-1]["n_epochs"]
        avg_train_loss = {key: np.mean(value) for key, value in self.epoch_history.loss.items()}
        avg_valid_metrics = {
            key: np.mean(value) for key, value in self.epoch_history.metrics.items()
        }
        self.history.update(
            loss=avg_train_loss,
            metrics=avg_valid_metrics,
            n_epochs=n_epochs + 1,
        )

    def perform_validation(self, valid_dataloader=None, verbose=True):
        if valid_dataloader:
            valid_metrics = self.validate(valid_dataloader)
            self.epoch_history.update(metrics=valid_metrics)
        if verbose:
            print(self.epoch_history)

    def checkpoint(self, model=None, optimizer=None, scheduler=None, condition=None, verbose=True):
        if callable(condition):
            in_condition = condition(self.history, self.epoch_history)
        else:
            in_condition = not condition

        if self.checkpoint_dir and in_condition:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

            if model:
                tag = list(model.keys())[0] if isinstance(model, dict) else "model"
                path = list(model.values())[0] if isinstance(model, dict) else model
                self.save(model=Path(self.checkpoint_dir) / Path(path), verbose=verbose)
                self.history.update(**{tag: path})

            if optimizer:
                tag = list(optimizer.keys())[0] if isinstance(optimizer, dict) else "optimizer"
                path = list(optimizer.values())[0] if isinstance(optimizer, dict) else optimizer
                self.save(model=Path(self.checkpoint_dir) / Path(path), verbose=verbose)
                self.history.update(**{tag: path})

            if scheduler and self.scheduler:
                tag = list(scheduler.keys())[0] if isinstance(scheduler, dict) else "scheduler"
                path = list(scheduler.values())[0] if isinstance(scheduler, dict) else scheduler
                self.save(model=Path(self.checkpoint_dir) / Path(path), verbose=verbose)
                self.history.update(**{tag: path})

            self.history.save(Path(self.checkpoint_dir) / "history.json")

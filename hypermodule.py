import torch
import numpy as np
from torch.nn.functional import softmax
from tqdm import tqdm
from functools import partial
from ..utils.partials import optim, sched


class HyperModule:
    def __init__(self, model, criterion, optimizer, scheduler=None, hyperparams=None):
        self._optimizer, self._scheduler, self._hyperparams = None, None, None
        self.model, self.criterion = model, criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hyperparams = hyperparams
        self.train_loss, self.valid_loss, self.valid_acc = [], [], []
        self.epoch_trained = 0
        self.test_acc = None
        self.load_path = None

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

    # ----------------------- train() --------------------------------------- #

    def train(
        self,
        train_dataloader,
        valid_dataloader=None,
        valid_loss=None,
        save_path=None,
        num_epochs=1,
        verbose=True,
    ):
        device = torch.device("cuda")
        self.model.to(device)
        self.load_path = save_path
        max_acc = 0 if len(self.valid_acc) == 0 else max(self.valid_acc)
        min_loss = np.inf if len(self.valid_loss) == 0 else min(self.valid_loss)
        start_epoch = self.epoch_trained

        # Train loop
        for epoch in range(num_epochs):
            self.batch = {
                "train_loss": [],
                "avg_train_loss": None,
                "valid_loss": [],
                "avg_valid_loss": None,
                "valid_acc": [],
                "avg_valid_acc": None,
            }
            self.batch_avg_valid_loss = None
            self.batch_avg_valid_acc = None
            self.batch["train_loss"], self.batch["valid_acc"] = [], []
            self.epoch_trained += 1

            # Training stage
            self.model.train()
            train_progress = tqdm(train_dataloader, position=0, leave=verbose)
            for images, targets in train_progress:
                images, targets = images.to(device), targets.to(device)
                self._update(images, targets)
                self._update_progress(train_progress, start_epoch + num_epochs)

            # Validatiing stage
            self._perform_validation(valid_dataloader, valid_loss, verbose)
            self._update_history()
            self.save(save_path, verbose=False)
            self._update_scheduler()

            # Save the best model (if any)
            if self.batch["avg_valid_loss"] < min_loss:
                min_loss = self.batch["avg_valid_loss"]
                self.save(save_path + ".best", verbose)

        # Clear training infomation
        self.batch["train_loss"], self.batch["valid_acc"] = [], []
        self.test_acc = None

    def _update(self, images, targets):
        preds = self.model(images)
        loss = self.criterion(preds, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.batch["train_loss"].append(loss.detach().item())

    def _update_progress(self, progress, num_epochs):
        loss = self.batch["train_loss"][-1]
        epoch = self.epoch_trained
        progress.set_description(f"Epoch [{epoch}/{num_epochs}]")
        if (
            self._scheduler is None
            or getattr(self._scheduler, "_last_lr", None) is None
        ):
            progress.set_postfix({"loss": loss})
        else:
            progress.set_postfix({"loss": loss, "lr": self._scheduler._last_lr})

    def _update_scheduler(self):
        if self._scheduler is not None:
            if "metrics" in self._scheduler.step.__code__.co_varnames:
                self._scheduler.step(self.batch["avg_valid_loss"])
            else:
                self._scheduler.step()

    def _update_history(self):
        self.train_loss.append(self.batch["avg_train_loss"])
        self.valid_loss.append(self.batch["avg_valid_loss"])
        self.valid_acc.append(self.batch["avg_valid_acc"])
        self.batch["train_loss"] = []
        self.batch["valid_loss"] = []
        self.batch["valid_acc"] = []

    def _perform_validation(self, valid_dataloader, valid_loss=None, verbose=True):
        if valid_dataloader is None:
            self.batch["valid_loss"] = [np.nan]
            self.batch["valid_acc"] = [np.nan]
        else:
            valid_loss, valid_acc = self.validate(valid_dataloader, valid_loss)
            self.batch["valid_loss"] = valid_loss
            self.batch["valid_acc"] = valid_acc

        self.batch["avg_train_loss"] = np.mean(self.batch["train_loss"])
        self.batch["avg_valid_loss"] = np.mean(self.batch["valid_loss"])
        self.batch["avg_valid_acc"] = np.mean(self.batch["valid_acc"])
        if verbose:
            print(
                f'Train Loss: {self.batch["avg_train_loss"]:.3f} ',
                f'Valid Loss: {self.batch["avg_valid_loss"]:.3f} ',
                f'Valid Acc: {self.batch["avg_valid_acc"]:.3f}',
            )

    # ------------------------ predict() ------------------------------------- #

    def predict(self, images=None, dataloader=None, numpy=False):
        device = torch.device("cuda")
        self.model.to(device)
        self.model.eval()
        if images is not None:
            images = images.to(device)
            return self._predict_image(images, numpy=numpy)
        if dataloader is not None:
            return self._predict_dataloader(dataloader, numpy=numpy)

    def _torch2numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def _predict_image(self, images, numpy=False):
        logits = self.model(images)
        probs = softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        if numpy:
            return self._torch2numpy(pred_labels)
        else:
            return pred_labels

    def _predict_dataloader(self, dataloader, return_target=False, numpy=False):
        device = torch.device("cuda")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            pred_list = []
            target_list = []
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                pred_labels = self._predict_image(images, numpy=numpy)
                pred_list.append(pred_labels)
                targets = self._torch2numpy(targets) if numpy else targets
                target_list.append(targets)

        output = (pred_list, target_list) if return_target else pred_list
        return output

    # ----------------------- validate() ------------------------------------- #

    def validate(self, dataloader, loss=None):
        device = torch.device("cuda")
        self.model.to(device)
        self.model.eval()
        valid_acc, valid_loss = [], []
        if loss is None:
            loss = self.criterion
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                logits = self.model(images)
                probs = softmax(logits, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
                valid_loss.append(loss(logits, targets).item())
                valid_acc.append(
                    (pred_labels == targets).type(torch.float32).mean().item()
                )
        return valid_loss, valid_acc

    # ------------------------ test() ---------------------------------------- #

    def test(self, dataloader, load_path="last", metrics=None, verbose=True, **kwargs):
        device = torch.device("cuda")
        self.model.to(device)
        if load_path == "last":
            self.load(self.load_path, verbose)
        elif load_path == "best":
            self.load(self.load_path + ".best", verbose)
        else:
            self.load(load_path, verbose)
        self.model.eval()

        # Obtain predictions and ground truths
        pred_list, target_list = self._predict_dataloader(
            dataloader, return_target=True, numpy=True
        )
        pred_labels = np.concatenate(pred_list)
        targets = np.concatenate(target_list)

        def default_metrics(targets, pred_labels):
            total_acc = np.mean(targets == pred_labels)
            return {"total_acc": total_acc}

        if metrics is None:
            self.test_acc = default_metrics(targets, pred_labels)
        else:
            self.test_acc = metrics(targets, pred_labels, **kwargs)

        return self.test_acc

    # ----------------------- load() ----------------------------------------- #

    def load(self, path, verbose=True):
        device = torch.device("cuda")
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict["model"])
        self.model.to(device)

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

    # ----------------------- save() ----------------------------------------- #

    def save(self, path, verbose=True):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        state_dict["optimizer"] = self._optimizer.state_dict()
        if self._scheduler is not None:
            state_dict["scheduler"] = self._scheduler.state_dict()
        else:
            state_dict["scheduler"] = None
        state_dict["epoch_trained"] = self.epoch_trained
        state_dict["train_loss"] = self.train_loss
        state_dict["valid_acc"] = self.valid_acc
        state_dict["test_acc"] = self.test_acc
        torch.save(state_dict, path)
        if verbose:
            print("State dict saved.")
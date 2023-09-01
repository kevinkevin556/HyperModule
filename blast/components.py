import torch
import tqdm
import numpy as np
from torch.nn.functional import softmax


class BaseModule:
    pass


class Predictor(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, images=None, dataloader=None):
        self.model.to("cuda")
        self.model.eval()
        if images:
            images = images.to("cuda")
            return self.predict_image(images)
        if dataloader:
            return self.predict_dataloader(dataloader)

    def predict_image(self, images):
        self.model.to("cuda")
        self.model.eval()
        images = images.to("cuda")
        logits = self.model(images)
        probs = softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        return pred_labels
    
    def predict_dataloader(self, dataloader, return_target=False):
        self.model.to("cuda")
        self.model.eval()
        with torch.no_grad():
            pred_list, target_list = [], []
            for images, targets in dataloader:
                images, targets = images.to("cuda"), targets.to("cuda")
                pred_labels = self.predict_image(images)
                pred_list.append(pred_labels)
                target_list.append(targets)
        output = (pred_list, target_list) if return_target else pred_list
        return output



class Evaluator(Predictor):
    def __init__(self, hypermodule, metrics):
        self.model = hypermodule.model
        self.predict = hypermodule.predict
        loss = self.model.criterion
        self.metrics = {
            "accuracy": lambda pred, target: np.mean(pred==target),
            loss.__class__.anme: loss   
        }
        self.metrics.update(metrics)
    
    
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
                valid_acc.append((pred_labels == targets).type(torch.float32).mean().item())
        return valid_loss, valid_acc


    def test(self, test_dataloader, verbose=True,):
        raise NotImplementedError
    


    
    def test(self, dataloader, load_path="last", metrics=None, verbose=True, **kwargs):
        def default_metrics(targets, pred_labels):
            total_acc = np.mean(targets == pred_labels)
            return {"total_acc": total_acc}

        device = torch.device("cuda")
        self.model.to(device)
        self.load(load_path, verbose)
        self.model.eval()

        # Obtain predictions and ground truths
        pred_list, target_list = self._predict_dataloader(
            dataloader, return_target=True, numpy=True
        )
        pred_labels = np.concatenate(pred_list)
        targets = np.concatenate(target_list)
        if metrics is None:
            self.test_acc = default_metrics(targets, pred_labels)
        else:
            self.test_acc = metrics(targets, pred_labels, **kwargs)
        return self.test_acc



class Trainer(Evaluator):
    def __init__(self, hypermodule, validator):
        self.model = hypermodule.model
        self.criterion = hypermodule.criterion
        self.optimizer = hypermodule.optimizer
        self.scheduler = hypermodule.scheduler
        self.history = hypermodule.history

        self.validator = validator
        self.validate = validator.validate

    def fit(self,train_dataloader, valid_dataloader=None, num_epochs=1, verbose=True, **kwargs):
        self.model.to("cuda")
        self.history.new_training()

        for epoch in range(num_epochs):
            self.epoch_trained += 1
            self.epoch_history = EpochHistory(loss=self.criterion, metrics=self.validator.metrics)
            self.model.train()
            train_progress = tqdm(train_dataloader, position=0, leave=verbose)
            for images, targets in train_progress:
                images, targets = images.to("cuda"), targets.to("cuda")
                self.update(images, targets)
                self.update_progressbar(train_progress, num_epochs)
            self.perform_validation(valid_dataloader, verbose)
            self.update_scheduler()
            self.update_history()
            self.checkpoint("last.pth")
            self.checkpoint("best.pth", if_valid_loss_improve)
        self.model.eval()
        del self.epoch_history


    def update(self, images, targets, **kwargs):
        # back-propagate
        preds = self.model(images)
        loss = self.criterion(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record loss
        self.epoch_history.update(loss={loss.__class__.name: loss.detach().item()})


    def update_progresbar(self, progress, num_epochs):
        # Update progress bar description
        n_epoch = self.start_epoch + self.history.training[-1].n_epoch
        total_epochs = self.start_epoch + num_epochs
        progress.set_description(f"Epoch [{n_epoch}/{total_epochs}]")
        
        # Update progress bar postfix
        loss = self.epoch_history.loss[-1]
        lr = getattr(self.scheduler, "_last_lr", None)
        postfix = {"loss": loss, "lr": lr} if lr else {"loss": loss}
        progress.set_postfix(postfix)
        

    def update_scheduler(self):
        if self.scheduler is None:
            pass
        elif "metrics" in self.scheduler.step.__code__.co_varnames:
            self.scheduler.step(self.batch["avg_valid_loss"])
        else:
            self.scheduler.step()


    def update_history(self):
        avg_train_loss = {key:value.mean() for key, value in self.epoch_history.loss.items()}
        avg_valid_metrics = {key:value.mean() for key, value in self.epoch_history.metrics.items()}
        self.history.update(loss=avg_train_loss, metrics=avg_valid_metrics)


    def perform_validation(self, valid_dataloader=None, verbose=True):
        if valid_dataloader:
            valid_metrics = self.validator(valid_dataloader)
            self.epoch_history.update(metrics=valid_metrics)
        if verbose:
            print(self.epoch_history)
               

    def checkpoint(self, checkpoint_dir, condition):
        if condition(self.history, self.epoch_history) or (condition is None):
            self.hypermodule.save(checkpoint_dir) # TODO: better way to write this?




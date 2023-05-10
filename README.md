# HyperModule

`HyperModule` is a wrapper of functions for managing the training, validation, and testing processes of neural networks. With HyperModule, it is easier to monitor the progress of the training process by logging loss and validation accuracy. Additionally, HyperModule provides convenient functions for loading and saving pre-trained models, streamlining the entire process of working with neural networks.

## Usage

### Getting Started

A simpliest way to use hypermodule is to create instances of network, optimizer, and scheduler first and then
loading them with a hypermodule:

```Python
# Example 1.
model = NN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

hm = HyperModule(model, criterion, optimizer, scheduler)

hm.train(train_dataloader, valid_dataloader, save_path, num_epochs=100)
hm.test(test_dataloader)
```
### Partials

However, **a more recommended approach is to assign optimizer and scheduler by partial functions**,
which save you from chaining model parameters, optimizer, and learning rate scheduler on your own.

```Python
# Example 2.
from .partials import optim, sched

hm  = Hypermodule(
    model = NN(),
    criterion = torch.nn.CrossEntropyLoss(),
    optimizer = optim("SGD", lr=0.01, momentum=0.9),
    scheduler = sched("ExponentialLR", gamma=0.9)
)
```
This is equivalent to Example 1.

The partial function `optim`/`sched` can take an existed optimizer/scheduler as its argument,
generate another optimizer/scheduler instance with the new hyperparameters. 

```Python
# Example 3.
model = NeuralNetwork()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

hm  = Hypermodule(
    model = NN(),
    criterion = torch.nn.CrossEntropyLoss(),
    optimizer = optim(optimizer, lr=0.005, momentum=0.99),
    scheduler = sched(scheduler, gamma=0.11)
)
```
Now the optimizer used in training is a SGD with learning rate 0.005 and momentum 0.99,
and the scheduler is an ExponentialLR scheduler with gamma 0.11. 

### Hyperparameters

With partials, we can provide hyperparmeters in a dict apart from the optimizer and scheduler functions.

```Python
from .partials import optim, sched

hyperparams = {
  'lr': 0.01, 
  'momentum': 0.9,
  'gamma': 0.9
}

hm  = Hypermodule(
    model = NN(),
    criterion = torch.nn.CrossEntropyLoss(),
    optimizer = optim("SGD"),
    scheduler = sched("ExponentialLR"),
    hyperparams = hyperparams
)
```
Note this is equivalent to Example 1 and 2.



## Class structure

- `__init__`: The constructor of the class which takes the model, criterion, optimizer, scheduler, and hyperparameters as input arguments.
- `optimizer`: A property that returns the optimizer used for training the model.
    - `optimizer.setter`: A method that sets the optimizer used for training the model, either by accepting an instance of `torch.optim.Optimizer` or by creating an optimizer based on the provided optimizer configuration.
- `scheduler`: A property that returns the scheduler used for adjusting the learning rate of the optimizer during training.
    - `scheduler.setter`: A method that sets the scheduler used for adjusting the learning rate of the optimizer during training, either by accepting an instance of `torch.optim.lr_scheduler.LRScheduler` or by creating a scheduler based on the provided scheduler configuration.
- `hyperparams`: A property that returns the hyperparameters used for optimizing the model.
    - `hyperparams.setter`: A method that sets the hyperparameters used for optimizing the model, either by accepting a dictionary of hyperparameters or by creating a new optimizer and scheduler based on the provided hyperparameters.
- `train`: A method that trains the model using the provided training and validation data loaders for the specified number of epochs. It also saves the best model based on the lowest validation loss and returns the training and validation losses and accuracy.
    - `_update`: A method that updates the model parameters based on the provided input images and targets and the current loss.
    - `_update_progress`: A method that updates the training progress bar based on the current epoch and batch loss.
    - `_update_scheduler`: A method that updates the learning rate of the optimizer using the scheduler.
    - `_update_history`: A method that updates the training history with the current epoch's training and validation losses and accuracy.
    - `_perform_validation`: A method that validates the model using the provided validation data loader and loss function and returns the validation loss and accuracy.
- `validate`: Evaluates the model on the given dataloader using the given criterion and returns the loss and accuracy.
- `test`: Evaluates the model on the given test dataloader using the given criterion and returns the accuracy.
- `predict`: Returns the model's predictions for the given dataloader, optionally applying softmax to the output.
- `save`: Saves the model and training information to the given save path, or to the HyperModule's load_path attribute if none is given.


## Loading and Saving

The *information* being loaded and saved in `HyperModule` is

* `state_dict` of neural network
* `state_dict` of optimizer
* `state_dict` of scheduler
* number of epochs that neural network has been trained
* training loss in each epoch
* validation accuracy in each epoch
* testing accuracy

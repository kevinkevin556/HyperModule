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

* `__init__()`: Assign neural network, optimizer, loss function and scheduler to a hypermodule instance
* `train()`
  * `update_()`: Conduct forward path and backward proporgation
  * `update_progress_()`: Update epoch information for `tqdm`
  * `update_scheduler_()`: Update scheduler
  * `update_history_()`: log training loss and validation accuracy.
  * `perform_validation_()`: perform validation by calling `valudate()` if `valid_dataloader` is not None.
* `validate()`
* `test()`: Test the performance of neural network
  * `get_prediction_()`: get predicted labels and ground truth of each batch
  * `visualize_class_acc_`: plot confusion heatmap and print testing accuracy of each class
    * `generate_confusion_df_()`: generate pandas `DataFrame` of confusion matrix
    * `plot_confusion_heatmap_()`: plot confusion heatmap
    * `print_class_acc_`: print accuracy of each class
* `load()`: read *information* of neural network from the given path.
* `save()`: save *information* to the given path.

## Loading and Saving

The *information* being loaded and saved in `HyperModule` is

* `state_dict` of neural network
* `state_dict` of optimizer
* `state_dict` of scheduler
* number of epochs that neural network has been trained
* training loss in each epoch
* validation accuracy in each epoch
* testing accuracy

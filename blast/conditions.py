import numpy as np


def if_valid_loss_improved(history, epoch_history):
    train_loss_name = list(history.training[-1]["loss"].keys())[0]
    valid_loss_values = history.training[-1]["metrics"][train_loss_name]
    if valid_loss_values[-1] == np.min(valid_loss_values):
        return True
    else:
        return False

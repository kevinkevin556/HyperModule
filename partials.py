import inspect
from functools import partial
import torch.optim
import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Type


def fix_default(gen: Type[Optimizer] | Type[LRScheduler], params: dict):
    """not all optimizer and lr_scheduler are fixed"""
    if gen == torch.optim.SGD:
        params["lr"] = 0.01
    if gen == torch.optim.lr_scheduler.ExponentialLR:
        params["gamma"] = 0.9
    return params


def optim(optimizer: Type[Optimizer] | Optimizer | str | partial, **hyperparam):
    """A partial function designed to generate optimizer using given hyperparameters.

    Args:
        optimizer (Type[Optimizer] | Optimizer | str | partial): what optimizer will be generated.

        * Type[Optimizer]: return a function that creates an optimizer of the given type using the hyperparameters.

        * Optimizer: return a function that generates a copy of the given optimizer. All defaults will be preserved,
        unlesss they are modified in the hyperparameter dict.

        * str: search the corresponding optimizer in torch.optim.

        * partial: simply return the input partial function with its keywords updated by hyperparameters.

    Raises:
        TypeError: If `optimizer` is not a string, instance or class related to optimizer or a partial.

    Returns:
        partial: A partial function to generate optimizer
    """

    is_optimizer_class = inspect.isclass(optimizer) and issubclass(
        optimizer, torch.optim.Optimizer
    )
    is_optimizer_instance = isinstance(optimizer, torch.optim.Optimizer)
    is_optimizer_str = isinstance(optimizer, str)
    is_optimizer = is_optimizer_class or is_optimizer_instance or is_optimizer_str

    if is_optimizer:
        if is_optimizer_instance:
            optim_gen = optimizer.__class__
            opt_params = optimizer.defaults
        else:
            optim_gen = (
                optimizer if is_optimizer_class else getattr(torch.optim, optimizer)
            )
            signature = inspect.signature(optim_gen.__init__)
            func_param_name = set(signature.parameters.keys())
            opt_params = {
                key: value.default
                for key, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty
            }
            opt_params = fix_default(optim_gen, opt_params)

        func_param_name = set(optim_gen.__init__.__code__.co_varnames)
        hyperparam_name = set(hyperparam.keys())
        opt_params.update(
            {var: hyperparam[var] for var in func_param_name & hyperparam_name}
        )
        output = partial(optim_gen, **opt_params)
        return output

    if isinstance(optimizer, partial):
        func_param_name = set(optimizer.keywords.keys())
        hyperparam_name = set(hyperparam.keys())
        opt_params = {var: hyperparam[var] for var in func_param_name & hyperparam_name}
        optimizer.keywords.update(opt_params)
        return optimizer

    else:
        raise TypeError(
            f"Invalid type for argument 'optimizer' in partial function 'optim', get type {type(optimizer)}"
        )


def sched(scheduler: Type[LRScheduler] | LRScheduler | str | partial, **hyperparam):
    is_scheduler_class = inspect.isclass(scheduler) and issubclass(
        scheduler, torch.optim.lr_scheduler.LRScheduler
    )
    is_scheduler_instance = isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
    is_scheduler_str = isinstance(scheduler, str)
    is_scheduler = is_scheduler_class or is_scheduler_instance or is_scheduler_str

    if is_scheduler:
        if is_scheduler_instance:
            sched_gen = scheduler.__class__
            func_param_name = set(sched_gen.__init__.__code__.co_varnames)
            sch_params = {
                key: getattr(scheduler, key)
                for key in func_param_name
                if key not in ["self", "optimizer"]
            }
            sch_params["last_epoch"] = -1
        else:
            sched_gen = (
                scheduler
                if is_scheduler_class
                else getattr(torch.optim.lr_scheduler, scheduler)
            )
            signature = inspect.signature(sched_gen.__init__)
            func_param_name = set(signature.parameters.keys())
            sch_params = {
                key: value.default
                for key, value in signature.parameters.items()
                if value.default is not inspect.Parameter.empty
            }
            sch_params = fix_default(sched_gen, sch_params)

        hyperparam_name = set(hyperparam.keys())
        sch_params.update(
            {var: hyperparam[var] for var in func_param_name & hyperparam_name}
        )
        output = partial(sched_gen, **sch_params)
        return output

    if isinstance(scheduler, partial):
        func_param_name = set(scheduler.keywords.keys())
        hyperparam_name = set(hyperparam.keys())
        sch_params = {var: hyperparam[var] for var in func_param_name & hyperparam_name}
        scheduler.keywords.update(sch_params)
        return scheduler

    else:
        raise TypeError(
            f"Invalid type for argument 'scheduler' in partial function 'sched', get type {type(scheduler)}"
        )

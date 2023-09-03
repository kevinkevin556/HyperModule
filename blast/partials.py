import inspect
from inspect import Parameter
from functools import partial
import torch.optim
import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Type, Union

try:
    import torch_optimizer
    torch_optimizer_installed = True
except ImportError:
    torch_optimizer_installed = False 


def fix_default(gen: Union[Type[Optimizer], LRScheduler], params: dict):
    """not all optimizer and lr_scheduler are fixed"""
    if gen == torch.optim.SGD:
        params["lr"] = 0.01
    if gen == torch.optim.lr_scheduler.ExponentialLR:
        params["gamma"] = 0.9
    return params


def optim(optimizer: Union[Type[Optimizer], Optimizer, str, partial], **hyperparam):
    """A partial function designed to generate optimizer using given hyperparameters.

    Args:
        optimizer (Type[Optimizer] | Optimizer | str | partial): optimizer to be generated.

        * Type[Optimizer]: return a function that creates an optimizer of the given type using the hyperparameters.

        * Optimizer: return a function that generates a copy of the given optimizer. All defaults will be preserved,
        unless they are modified in the hyperparameter dict.

        * str: search the corresponding optimizer in torch.optim (and torch_optimizer if this library is installed).

        * partial: simply return the input partial function with its keywords updated by hyperparameters.

    Raises:
        TypeError: If `optimizer` is not a string, optimizer instance, optimizer class or a partial.

    Returns:
        partial: A partial function to generate optimizer
    """

    if isinstance(optimizer, partial):
        optim_partial = optimizer
        init_param_name = set(optimizer.keywords.keys())
        hyperparam_name = set(hyperparam.keys())
        opt_params = {var: hyperparam[var] for var in init_param_name & hyperparam_name}
        optim_partial.keywords.update(opt_params)
    else:
        # Obtain optimizer class
        if inspect.isclass(optimizer) and issubclass(optimizer, Optimizer):
            optim_class = optimizer
        elif isinstance(optimizer, Optimizer):
            optim_class = optimizer.__class__
        elif isinstance(optimizer, str):
            if torch_optimizer_installed:
                optim_class = getattr(torch_optimizer, optimizer, None)
            if (not torch_optimizer_installed) or (optim_class is None):
                optim_class = getattr(torch.optim, optimizer)
        else:
            raise TypeError(
                f"Invalid type for argument 'optimizer' in partial function 'optim', get type {type(optimizer)}"
            )

        # Get optimizer parameters from its signature
        signature = inspect.signature(optim_class.__init__)
        optim_init_param = signature.parameters
        if isinstance(optimizer, Optimizer):
            p_keys = optim_init_param.keys()
            opt_params = {
                k: optimizer.defaults[k] for k in p_keys if k not in ("self", "params")
            }
        else:
            p_items = optim_init_param.items()
            opt_params = {
                k: v.default for k, v in p_items if v.default is not Parameter.empty
            }
        opt_params = fix_default(optim_class, opt_params)

        # Update optimizer parameters if hyperparams are provided
        init_param_name = set(optim_init_param.keys())
        hyperparam_name = set(hyperparam.keys())
        opt_params.update(
            {var: hyperparam[var] for var in init_param_name & hyperparam_name}
        )
        optim_partial = partial(optim_class, **opt_params)
    return optim_partial


def sched(scheduler: Union[Type[LRScheduler], LRScheduler, str, partial], **hyperparam):
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

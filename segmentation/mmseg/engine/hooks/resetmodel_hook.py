import torch
from spikingjelly.clock_driven import functional
from typing import Optional, Sequence
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class ResetModelHook(Hook):
    """
    This hook is used for reset the network when train and inference in SNN mode
    We reset the model after each iteration
    """
    def __init__(self, **kwargs):
        super(ResetModelHook, self).__init__(
             **kwargs)

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        torch.cuda.synchronize()
        functional.reset_net(runner.model)

    def before_val_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        torch.cuda.synchronize()
        functional.reset_net(runner.model)

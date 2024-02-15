import torch
from spikingjelly.clock_driven import functional
from typing import Optional, Sequence
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class Get_lif_firing_num(Hook):
    """Docstring for NewHook.
    """
    def __init__(self, **kwargs):
        super(Get_lif_firing_num, self).__init__(
             **kwargs)

    # def before_train_iter(self,
    #                       runner,
    #                       batch_idx: int,
    #                       data_batch: Optional[Sequence[dict]] = None) -> None:
    #     import pdb; pdb.set_trace()

    # def after_train_iter(self,
    #                       runner,
    #                       batch_idx: int,
    #                       outputs: None,
    #                       data_batch: Optional[Sequence[dict]] = None) -> None:
    #     torch.cuda.synchronize()
    #     functional.reset_net(runner.model)

    def before_test_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        import pdb; pdb.set_trace()

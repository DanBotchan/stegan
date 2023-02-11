from typing import NamedTuple

import torch
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt


def show_tensor_image(tensor: torch.Tensor, range_zero_one: bool = False):
    """Show a tensor of an image

    Args:
        tensor (torch.Tensor): Tensor of shape [N, 3, H, W] in range [-1, 1] or in range [0, 1]
    """
    if not range_zero_one:
        tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)

    batch_size = tensor.shape[0]
    for i in range(batch_size):
        plt.title(f"Fig_{i}")
        pil_image = TF.to_pil_image(tensor[i])
        plt.imshow(pil_image)
        plt.show(block=True)

class BaseReturn(NamedTuple):
    encoded: torch.Tensor
    decoded: torch.Tensor = None


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup

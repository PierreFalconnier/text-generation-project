import torch
import inspect
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# SCHEDULER


class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self._get_lr(it) for it in self.base_lrs]

    def _get_lr(self, base_lr):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            return base_lr * self.last_epoch / self.warmup_iters
        elif self.last_epoch > self.lr_decay_iters:
            # Return minimum learning rate
            return self.min_lr
        else:
            # Cosine decay
            decay_ratio = (self.last_epoch - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + coeff * (base_lr - self.min_lr)


# OPTIMIZER


# CODE FROM https://github.com/karpathy/nanoGPT
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer

"""
Manual AdamW optimizer for Project 2.

This class intentionally does not subclass or call torch.optim. It uses
autograd-computed gradients and performs the full AdamW parameter update by
hand, which matches the assignment's optimizer requirement while keeping the
model itself in ordinary torch.nn modules.
"""
import math

import torch


class ManualAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")

        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("ManualAdamW received no trainable parameters")
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas

        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("ManualAdamW does not support sparse gradients")

            state = self.state.get(param)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                }
                self.state[param] = state

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if self.weight_decay != 0:
                param.mul_(1 - self.lr * self.weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self):
        param_to_index = {param: index for index, param in enumerate(self.params)}
        state = {}
        for param, values in self.state.items():
            state[param_to_index[param]] = {
                "step": values["step"],
                "exp_avg": values["exp_avg"].clone(),
                "exp_avg_sq": values["exp_avg_sq"].clone(),
            }
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": state,
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict["lr"]
        self.betas = tuple(state_dict["betas"])
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]

        self.state = {}
        for index, values in state_dict["state"].items():
            param = self.params[int(index)]
            self.state[param] = {
                "step": values["step"],
                "exp_avg": values["exp_avg"].to(param.device),
                "exp_avg_sq": values["exp_avg_sq"].to(param.device),
            }

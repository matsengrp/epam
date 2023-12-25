import math

import numpy as np
import torch
import torch.optim as optim
from torch import nn, Tensor


def pick_device():
    # check that CUDA is usable
    def check_CUDA():
        try:
            torch._C._cuda_init()
            return True
        except:
            return False

    if torch.backends.cudnn.is_available() and check_CUDA():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def optimize_branch_length(
    log_prob_fn,
    starting_branch_length,
    learning_rate=0.1,
    max_optimization_steps=1000,
    optimization_tol=1e-3,
    log_branch_length_lower_threshold=-10.0,
):
    log_branch_length = torch.tensor(np.log(starting_branch_length), requires_grad=True)

    optimizer = optim.Adam([log_branch_length], lr=learning_rate)
    prev_log_branch_length = log_branch_length.clone()

    step_idx = 0

    for step_idx in range(max_optimization_steps):
        # For some PCPs, the optimizer works very hard optimizing very tiny branch lengths.
        if log_branch_length < log_branch_length_lower_threshold:
            break

        optimizer.zero_grad()

        loss = -log_prob_fn(log_branch_length)
        assert not torch.isnan(
            loss
        ), "Loss is NaN: perhaps selection has given a probability of zero?"
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_branch_length], max_norm=5.0)
        optimizer.step()
        assert not torch.isnan(log_branch_length)

        change_in_log_branch_length = torch.abs(
            log_branch_length - prev_log_branch_length
        )
        if change_in_log_branch_length < optimization_tol:
            break

        prev_log_branch_length = log_branch_length.clone()

    if step_idx == max_optimization_steps - 1:
        print(
            f"Warning: optimization did not converge after {max_optimization_steps} steps; log branch length is {log_branch_length.detach().item()}"
        )

    return torch.exp(log_branch_length.detach()).item()

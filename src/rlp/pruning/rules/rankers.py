"""Channel ranking utilities used by the pruning pipeline."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def l2_scores_conv2d(m: nn.Conv2d, conv_type: Optional[str]) -> torch.Tensor:
    weight = m.weight.data
    return weight.view(weight.size(0), -1).pow(2).sum(dim=1)


def l2_scores_linear(m: nn.Linear) -> torch.Tensor:
    weight = m.weight.data
    return weight.pow(2).sum(dim=1)


def select_keep_idx(scores: torch.Tensor, k: int) -> List[int]:
    k = max(1, min(k, scores.numel()))
    _, idx = torch.topk(scores, k, largest=True, sorted=True)
    idx, _ = torch.sort(idx)
    return idx.tolist()


__all__ = ["l2_scores_conv2d", "l2_scores_linear", "select_keep_idx"]

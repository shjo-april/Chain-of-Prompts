# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backbones from the TIMM library."""

from typing import List, Tuple

import torch
from torch import nn

from .timm_utils import build_repvit_with_cfg

class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        features: Tuple[str, ...],
    ):
        super().__init__()

        out_indices = tuple(int(f[len("layer") :]) for f in features)

        backbone = build_repvit_with_cfg(
            name,
            features_only=True,
            out_indices=out_indices,
            legacy=True
        )

        self.channel_list = backbone.feature_info.channels()[::-1]
        self.body = backbone

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.body(x)
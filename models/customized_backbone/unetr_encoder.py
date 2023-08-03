import torch.nn as nn
import torch.nn.functional as F

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep



class UNETR_encoder(nn.Module):
    """
    This is a template for defining your customized 3D backbone and use it for pre-training in ULIP framework.
    The expected input is Batch_size x num_points x 3, and the expected output is Batch_size x point_cloud_feat_dim
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        # feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        # norm_name: tuple | str = "instance",
        # conv_block: bool = True,
        # res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        # Initialization code...
       
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        # A new linear layer to produce the final class predictions.
        # self.class_token_fc = nn.Linear(hidden_size, out_channels)
        # return x, hidden_states_out

    def forward(self, x_in):
        # x, _ = self.vit(x_in)
        _, embedding = self.vit(x_in)
        # The classification token is used as the feature for classification.
        # x = self.class_token_fc(x)
        return embedding

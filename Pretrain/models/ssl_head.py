# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae"):
        super().__init__()
        self.spatial_dims = args.spatial_dims  # 2 or 3

        # Backbone
        patch_size  = ensure_tuple_rep(2, self.spatial_dims)
        window_size = ensure_tuple_rep(7, self.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[1, 1, 1, 1],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=self.spatial_dims,
        )

        # Feature dim of last stage = embed_dim * 2**(num_stages-1)
        num_stages = 4
        dim = args.feature_size * 16  # e.g., 48 -> 768

        # Heads
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)

        # Decoder / reconstruction head (2D or 3D)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, (32, 32, 32), (32, 32, 32)) \
                if self.spatial_dims == 3 else \
                nn.ConvTranspose2d(dim, args.in_channels, (32, 32), (32, 32))

        elif upsample == "deconv":
            if self.spatial_dims == 3:
                self.conv = nn.Sequential(
                    nn.ConvTranspose3d(dim,       dim // 2, (2, 2, 2), (2, 2, 2)),
                    nn.ConvTranspose3d(dim // 2,  dim // 4, (2, 2, 2), (2, 2, 2)),
                    nn.ConvTranspose3d(dim // 4,  dim // 8, (2, 2, 2), (2, 2, 2)),
                    nn.ConvTranspose3d(dim // 8,  dim // 16,(2, 2, 2), (2, 2, 2)),
                    nn.ConvTranspose3d(dim // 16, args.in_channels, (2, 2, 2), (2, 2, 2)),
                )
            else:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(dim,       dim // 2, 2, 2),
                    nn.ConvTranspose2d(dim // 2,  dim // 4, 2, 2),
                    nn.ConvTranspose2d(dim // 4,  dim // 8, 2, 2),
                    nn.ConvTranspose2d(dim // 8,  dim // 16,2, 2),
                    nn.ConvTranspose2d(dim // 16, args.in_channels, 2, 2),
                )

        elif upsample == "vae":
            if self.spatial_dims == 3:
                self.conv = nn.Sequential(
                    nn.Conv3d(dim, dim // 2, 3, padding=1), nn.InstanceNorm3d(dim // 2), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(dim // 2, dim // 4, 3, padding=1), nn.InstanceNorm3d(dim // 4), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(dim // 4, dim // 8, 3, padding=1), nn.InstanceNorm3d(dim // 8), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(dim // 8, dim // 16,3, padding=1), nn.InstanceNorm3d(dim // 16), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(dim // 16, dim // 16,3, padding=1), nn.InstanceNorm3d(dim // 16), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(dim // 16, args.in_channels, 1),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(dim, dim // 2, 3, padding=1), nn.InstanceNorm2d(dim // 2), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim // 2, dim // 4, 3, padding=1), nn.InstanceNorm2d(dim // 4), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim // 4, dim // 8, 3, padding=1), nn.InstanceNorm2d(dim // 8), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim // 8, dim // 16,3, padding=1), nn.InstanceNorm2d(dim // 16), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim // 16, dim // 16,3, padding=1), nn.InstanceNorm2d(dim // 16), nn.LeakyReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim // 16, args.in_channels, 1),
                )
        else:
            raise ValueError(f"Unknown upsample mode: {upsample}")

    def forward(self, x):
        # x_out is (B, C, H/32, W/32[, D/32]) for last stage
        out = self.swinViT(x.contiguous())
        x_out = out[-1]  # use last stage regardless of list length

        # token features for heads: (B, N, C), works for 2D or 3D
        tokens = x_out.flatten(2).transpose(1, 2)
        x_rot = self.rotation_head(self.rotation_pre(tokens[:, 0]))
        x_contrastive = self.contrastive_head(self.contrastive_pre(tokens[:, 1]))

        # reconstruction from feature map (keep shape, decode per spatial_dims)
        x_rec = self.conv(x_out)
        return x_rot, x_contrastive, x_rec


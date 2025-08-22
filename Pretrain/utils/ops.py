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

import numpy as np
import torch
from numpy.random import randint


def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    """
    Random block dropout compatible with 2D (C,H,W) and 3D (C,H,W,Z).
    """
    dev = x.device

    if x.dim() == 3:
        c, h, w = x.size()
        n_drop_pix = np.random.uniform(0, max_drop) * h * w
        mx_blk_height = int(h * max_block_sz)
        mx_blk_width  = int(w * max_block_sz)
        tol_h = int(tolr * h)
        tol_w = int(tolr * w)
        total_pix = 0
        while total_pix < n_drop_pix:
            rnd_r = randint(0, max(1, h - tol_h))
            rnd_c = randint(0, max(1, w - tol_w))
            rnd_h = min(randint(max(1, tol_h), max(2, mx_blk_height)) + rnd_r, h)
            rnd_w = min(randint(max(1, tol_w), max(2, mx_blk_width))  + rnd_c, w)
            if x_rep is None:
                patch = torch.empty((c, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=dev).normal_()
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                x[:, rnd_r:rnd_h, rnd_c:rnd_w] = patch
            else:
                x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]
            total_pix += (rnd_h - rnd_r) * (rnd_w - rnd_c)
        return x

    elif x.dim() == 4:
        c, h, w, z = x.size()
        n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
        mx_blk_height = int(h * max_block_sz)
        mx_blk_width  = int(w * max_block_sz)
        mx_blk_slices = int(z * max_block_sz)
        tol = (int(tolr * h), int(tolr * w), int(tolr * z))
        total_pix = 0
        while total_pix < n_drop_pix:
            rnd_r = randint(0, max(1, h - tol[0]))
            rnd_c = randint(0, max(1, w - tol[1]))
            rnd_s = randint(0, max(1, z - tol[2]))
            rnd_h = min(randint(max(1, tol[0]), max(2, mx_blk_height)) + rnd_r, h)
            rnd_w = min(randint(max(1, tol[1]), max(2, mx_blk_width))  + rnd_c, w)
            rnd_z = min(randint(max(1, tol[2]), max(2, mx_blk_slices)) + rnd_s, z)
            if x_rep is None:
                patch = torch.empty((c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=dev).normal_()
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = patch
            else:
                x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
            total_pix += (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
        return x

    else:
        raise ValueError(f"Unexpected tensor shape {x.shape} in patch_rand_drop")



def rot_rand(args, x_s):
    """
    x_s: (N, C, H, W) for 2D  OR  (N, C, H, W, Z) for 3D
    Rotates in the image plane (H, W) regardless of 2D/3D.
    """
    img_n = x_s.size(0)
    x_aug = x_s.detach().clone()
    dev = x_s.device  # match input device
    x_rot = torch.zeros(img_n, dtype=torch.long, device=dev)

    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation != 0:
            if x.dim() == 3:          # (C, H, W)  -> rotate over (H, W)
                x = torch.rot90(x, orientation, dims=(1, 2))
            elif x.dim() == 4:        # (C, H, W, Z) -> rotate each slice in (H, W)
                # rotate the whole volume in H-W plane
                x = torch.rot90(x, orientation, dims=(1, 2))
            else:
                raise ValueError(f"Unexpected sample dims: {x.shape}")
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot



def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug

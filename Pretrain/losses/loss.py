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
from torch.nn import functional as F

class Contrast(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        # keep as a buffer, but don't pin to cuda here
        self.register_buffer("temp", torch.tensor(float(temperature)))

    def forward(self, x_i, x_j):
        """
        x_i, x_j: (B, D) on the SAME device (cpu or cuda)
        """
        device = x_i.device
        temp = self.temp.to(device)

        # normalize and concat
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)              # (2B, D)

        # cosine similarities
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B, 2B)

        B = x_i.size(0)
        sim_ij = torch.diag(sim,  B)   # positives: i vs j
        sim_ji = torch.diag(sim, -B)
        pos = torch.cat([sim_ij, sim_ji], dim=0)      # (2B,)

        # masks built on-the-fly on correct device
        N = 2 * B
        neg_mask = (~torch.eye(N, N, dtype=bool, device=device)).float()

        nom = torch.exp(pos / temp)                   # (2B,)
        denom = neg_mask * torch.exp(sim / temp)      # (2B, 2B)
        loss = -torch.log(nom / denom.sum(dim=1))     # (2B,)
        return loss.mean()


class Loss(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        # don't pin to cuda; these are parameterless
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = Contrast(temperature=0.5)
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        # everything should already be on the same device as the model outputs
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss
        return total_loss, (rot_loss, contrast_loss, recon_loss)

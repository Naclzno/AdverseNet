
import torch, torch.nn as nn, torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from copy import deepcopy
from utils.lovasz_losses import lovasz_softmax
from utils.sem_geo_loss import geo_scal_loss, sem_scal_loss
import numpy as np


@MODELS.register_module()
class TPVAggregator_Seg(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=5, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False
    ):
        super().__init__()
        self.tpv_h = tpv_h # 180
        self.tpv_w = tpv_w # 240
        self.tpv_z = tpv_z # 16
        self.scale_h = scale_h # 2
        self.scale_w = scale_w # 2
        self.scale_z = scale_z # 2

        out_dims = in_dims if out_dims is None else out_dims # 192

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), # 192，384
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims) # 384，192
        )

        self.classifier = nn.Linear(out_dims, nbr_classes) # 192，5
        self.classes = nbr_classes # 5
        self.use_checkpoint = use_checkpoint

    def forward(self, tpv_list, points_list=None):
        """
        480 360 32
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2) # 1, 192, h, w
        tpv_zh = tpv_yz.permute(0, 1, 3, 2) # 1, 192, z, h
        tpv_wz = tpv_zx.permute(0, 1, 3, 2) # 1, 192, w, z

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)), # 360, 480
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)), # 32, 360
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)), # 480, 32
                mode='bilinear'
            )
        
        logits_pts_list = []

        for idx, points in enumerate(points_list):
            n, _ = points.shape
            points = points.reshape(1, 1, n, 3)
            points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_sample = tpv_hw[idx].unsqueeze(0)
            tpv_hw_pts = F.grid_sample(tpv_hw_sample, sample_loc, padding_mode="border").squeeze(2)

            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_sample = tpv_zh[idx].unsqueeze(0)
            tpv_zh_pts = F.grid_sample(tpv_zh_sample, sample_loc, padding_mode="border").squeeze(2)

            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_sample = tpv_wz[idx].unsqueeze(0) 
            tpv_wz_pts = F.grid_sample(tpv_wz_sample, sample_loc, padding_mode="border").squeeze(2)

            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused = fused_pts.permute(0, 2, 1)  # bs, n, c

            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)

            logits = logits.permute(0, 2, 1)  # bs, nbr_classes, n
            logits_pts = logits.reshape(1, self.classes, n, 1, 1)

            logits_pts_list.append(logits_pts)

        return logits_pts_list        
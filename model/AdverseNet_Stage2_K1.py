from mmengine.model import BaseModule
from mmengine.registry import build_from_cfg
from mmdet3d.registry import MODELS
import torch


@MODELS.register_module()
class AdverseNet_Stage2_K1(BaseModule):
    def __init__(self,
                 lidar_tokenizer=None, 
                 tpv_encoder_decoder=None,
                 tpv_aggregator=None, 
                 **kwargs,
                 ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = build_from_cfg(lidar_tokenizer, MODELS)
        if tpv_encoder_decoder:
            self.tpv_encoder_decoder = build_from_cfg(tpv_encoder_decoder, MODELS)
        if tpv_aggregator:
            self.tpv_aggregator = build_from_cfg(tpv_aggregator, MODELS)

        self.fp16_enabled = False

    def getIndicators_B1(self):
        return self.tpv_encoder_decoder.getIndicators_B1()
    
    def getIndicators_B2(self):
        return self.tpv_encoder_decoder.getIndicators_B2()
    
    def getIndicators_B3(self):
        return self.tpv_encoder_decoder.getIndicators_B3()

    def extract_lidar_feat(self, points, grid_ind, flag):
        """Extract features of points."""
        # x_3view = [tpv_xy, tpv_yz, tpv_zx]
        # tpv_xy = batch_size(1), channels(32), x(480), y(360)
        # tpv_yz = batch_size(1), channels(32), y(360), z(32)
        # tpv_zx = batch_size(1), channels(32), z(32), x(480)
        x_3view = self.lidar_tokenizer(points, grid_ind) 
        tpv_list = []
        for x in x_3view:
            x = self.tpv_encoder_decoder(x, flag)
            tpv_list.append(x) 
        return tpv_list 

    def forward(self,
                points=None, 
                grid_ind=None, 
                grid_ind_vox=None, 
                flag=None,
        ):
        """Forward training function.
        """
        x_lidar_tpv = self.extract_lidar_feat(points=points, grid_ind=grid_ind, flag=flag) 
        grid_ind_ = grid_ind_vox if grid_ind_vox is not None else grid_ind 
        outs = self.tpv_aggregator(x_lidar_tpv, grid_ind_) # TPVAggregator_Seg
        return outs
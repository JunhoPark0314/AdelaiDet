import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from .dcr_outputs import DCROutputs


__all__ = ["DCRv3"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class DCRv3(nn.Module):
    """
    Implement DCR (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.DCR.IN_FEATURES
        self.fpn_strides = cfg.MODEL.DCR.FPN_STRIDES

        self.dcr_head = DCRHead(cfg, {f:input_shape[f] for f in self.in_features})
        self.in_channels_to_top_module = self.dcr_head.in_channels_to_top_module

        self.dcr_outputs = DCROutputs(cfg)
        self.pos_sample_rate = nn.Parameter(torch.Tensor([cfg.MODEL.DCR.POS_SAMPLE_INIT]), requires_grad=False)
        self.pos_sample_limit = cfg.MODEL.DCR.POS_SAMPLE_LIMIT
        self.max_iter = cfg.SOLVER.MAX_ITER

    def iterate_disp(self, pred_disp):

        num_it = int((torch.max(self.pos_sample_rate + 0.5, 0)[0]).item() // 0.2)
        num_touch = [x.new_ones(x.shape[0],x.shape[2], x.shape[3]).long() for x in pred_disp]

        for i in range(num_it):
            for l, (disp_per_level, num_touch_per_level) in enumerate(zip(pred_disp, num_touch)):
                N, _, H, W  = disp_per_level.shape
                location = cat([disp_per_level.new_ones(H, W).nonzero().reshape(1, H, W, 2)] * N).permute(0,3,1,2)
                pred_cent = (location + disp_per_level).long()
                # Need clamp here
                pred_cent_flatten = pred_cent.permute(0,2,3,1).reshape(-1,2)
                pred_cent_flatten[:,0] = pred_cent_flatten[:,0].clamp(0,H-1)
                pred_cent_flatten[:,1] = pred_cent_flatten[:,1].clamp(0,W-1)
                pred_cent_flatten = cat([torch.arange(N, device=pred_cent.device).reshape(-1,1).repeat(1,H*W).reshape(-1,1), pred_cent_flatten], dim=1)

                assert torch.logical_and(pred_cent_flatten[:,1].max() < H, pred_cent_flatten[:,1].min() >= 0).item()
                assert torch.logical_and(pred_cent_flatten[:,2].max() < W, pred_cent_flatten[:,2].min() >= 0).item()

                unique_pred_cent, pred_count = pred_cent_flatten.unique(dim=0, return_counts=True)
                new_touch = torch.zeros_like(num_touch_per_level)
                new_touch[unique_pred_cent[:,0],unique_pred_cent[:,1], unique_pred_cent[:,2]] = pred_count

                inner_disp = disp_per_level[pred_cent_flatten[:,0], :, pred_cent_flatten[:,1], pred_cent_flatten[:,2]].reshape(N, H, W, 2).permute(0,3,1,2)
                pred_disp[l] = inner_disp + disp_per_level
                num_touch[l] = new_touch

        return pred_disp, num_touch
    
    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_target, pred_cls_ctr, pred_box_ctr = self.dcr_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_target, pred_cls_ctr,pred_box_ctr

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = {f:features[f] for f in self.in_features}
        locations = self.compute_locations(features)
        pred_result = self.dcr_head(features)
        #pred_result["pred_disp"], num_touch = self.iterate_disp(pred_result["pred_disp"])

        results = {}

        self.pos_sample_rate += (self.pos_sample_limit - self.pos_sample_rate) * 2 / self.max_iter
        self.dcr_outputs.pos_sample_rate = self.pos_sample_rate.item()

        if self.training:
            results, losses = self.dcr_outputs.losses(
                pred_result, locations, gt_instances, images.image_sizes
            )
            
            return results, losses
        else:
            #pred_result["num_touch"] = num_touch
            self.dcr_outputs.pos_sample_rate = 0.5
            training_target = None
            if len(gt_instances[0]):
                training_target = self.dcr_outputs._get_ground_truth(locations, gt_instances, pred_result, images.image_sizes)
            results, analysis = self.dcr_outputs.predict_proposals(
                pred_result, locations, images, 
                training_target = training_target if training_target is not None else None
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, (k, feature) in enumerate(features.items()):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class DCRHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str,ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.DCR.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.DCR.FPN_STRIDES
        self.head_configs = cfg.MODEL.DCR.HEAD_CFG
        norm = None if cfg.MODEL.DCR.NORM == "none" else cfg.MODEL.DCR.NORM
        self.num_levels = len(input_shape)

        num_convs, use_deformable = cfg.MODEL.DCR.NUM_CONVS, cfg.MODEL.DCR.USE_DEFORMABLE

        in_channels = [s.channels for s in list(input_shape.values())]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        head_per_lvl = {}

        for k in list(self.head_configs.keys()):

            tower = []

            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            
            head_per_lvl[k] = nn.Sequential(*tower)

        self.pred_cls = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.pred_iou = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.pred_disp = nn.Conv2d(
            in_channels, 2,
            kernel_size=3, stride=1,
            padding=1
        )

        if cfg.MODEL.DCR.USE_SCALE:
            self.scales = nn.ModuleDict({
                k: Scale(init_value=1.0) for k in list(input_shape.keys())
                })
        else:
            self.scales = None

        self.head_per_lvl = nn.ModuleDict(head_per_lvl)

        for modules in [
            self.head_per_lvl,
            self.pred_cls, self.pred_disp,
            self.bbox_pred,  self.pred_iou, 
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DCR.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.pred_cls.bias, bias_value)
        torch.nn.init.constant_(self.pred_iou.bias, bias_value)

    def forward(self, x):
        # logits
        pred_cls = []

        # bbox pred & iou pred
        pred_reg = []
        pred_iou = []

        # displacement vector
        pred_disp = []

        for (k, feature), (_, cfg_per_lvl) in zip(x.items(), self.head_configs.items()):

            feature = self.head_per_lvl[k](feature)
            
            for head in cfg_per_lvl['HEAD']:
                if 'CLS' == head:
                    pred_cls.append(self.pred_cls(feature))

                elif 'BBOX' == head:
                    reg = self.bbox_pred(feature)
                    iou = self.pred_iou(feature)

                    if self.scales is not None:
                        reg = self.scales[k](reg)

                    reg[:,2:] = reg[:,2:].exp()
                    pred_reg.append(reg)
                    pred_iou.append(iou)
                elif 'DISP' == head:
                    disp = self.pred_disp(feature)
                    pred_disp.append(disp)

        pred_result = {
            "pred_cls": pred_cls,
            "pred_reg": pred_reg,
            "pred_iou": pred_iou,
            "pred_disp": pred_disp,
        }

        return pred_result

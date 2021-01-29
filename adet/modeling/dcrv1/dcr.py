import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from detectron2.utils.events import get_event_storage
from .dcr_outputs import DCROutputs


__all__ = ["DCR"]

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
class DCR(nn.Module):
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

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_target = self.dcr_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_target

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
        logits_pred, reg_pred, pred_target = self.dcr_head(features)

        results = {}

        self.pos_sample_rate += (self.pos_sample_limit - self.pos_sample_rate) * 2 / self.max_iter
        self.dcr_outputs.pos_sample_rate = self.pos_sample_rate.item()

        if self.training:
            results, losses = self.dcr_outputs.losses(
                logits_pred, reg_pred, pred_target,
                locations, gt_instances
            )
            
            return results, losses
        else:
            cls_targets = None
            reg_targets = None
            if len(gt_instances[0]):
                cls_targets, reg_targets = self.dcr_outputs._get_ground_truth(locations, gt_instances,
                                                                    logits_pred, reg_pred, pred_target)
            results, analysis = self.dcr_outputs.predict_proposals(
                logits_pred, reg_pred, pred_target,
                locations, images.image_sizes, training_target = {"cls": cls_targets, "reg": reg_targets} if cls_targets is not None else None
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

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.target_anchor = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.DCR.USE_SCALE:
            self.scales = nn.ModuleDict({
                k: Scale(init_value=1.0) for k in list(input_shape.keys())
                })
        else:
            self.scales = None

        self.head_per_lvl = nn.ModuleDict(head_per_lvl)

        for modules in [
            self.head_per_lvl, self.cls_logits,
            self.bbox_pred, self.target_anchor
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DCR.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.target_anchor.bias, bias_value)


    def forward(self, x):
        logits = []
        bbox_reg = []
        trg_anchor = []

        for (k, feature), (_, cfg_per_lvl) in zip(x.items(), self.head_configs.items()):

            feature = self.head_per_lvl[k](feature)
            
            for head in cfg_per_lvl['HEAD']:
                if 'CLS' == head:
                    logits.append(self.cls_logits(feature))
                elif 'BBOX' == head:
                    reg = self.bbox_pred(feature)

                    if self.scales is not None:
                        reg = self.scales[k](reg)
                    
                    bbox_reg.append(reg.exp())

                elif 'TRG' == head:
                    trg_anchor.append(self.target_anchor(feature))

        return logits, bbox_reg, trg_anchor

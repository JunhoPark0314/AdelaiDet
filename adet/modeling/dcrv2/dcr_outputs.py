from collections import defaultdict
import logging
from detectron2.structures.boxes import matched_boxlist_iou
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms, IOULoss
from detectron2.structures import pairwise_iou

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class DCROutputs(nn.Module):
    def __init__(self, cfg):
        super(DCROutputs, self).__init__()

        self.focal_loss_alpha = cfg.MODEL.DCR.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.DCR.LOSS_GAMMA
        self.is_in_boxes = cfg.MODEL.DCR.IS_IN_BOXES
        self.radius = cfg.MODEL.DCR.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.DCR.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.DCR.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.DCR.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.DCR.LOC_LOSS_TYPE)
        self.iou_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.disp_loss_func = nn.SmoothL1Loss(reduction="sum")

        self.pre_nms_thresh_test = cfg.MODEL.DCR.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.DCR.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.DCR.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.DCR.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.DCR.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.DCR.NUM_CLASSES
        self.strides = cfg.MODEL.DCR.FPN_STRIDES
        self.instance_weight = cfg.MODEL.DCR.INSTANCE_WEIGHT


        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.DCR.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances, pred_result, image_size):

        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_target = self.compute_DCR_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list, pred_result, image_size
        )

        # transpose im first training_targets to level first ones
    
        for trg, trg_target in training_target.items():
            training_target[trg] = {
                k: self._transpose(v, num_loc_list) for k, v in trg_target.items()
            }   

        def get_angle(ctr, stride):
            v_len = torch.linalg.norm(ctr[l], dim=1) / stride
            base = torch.zeros_like(ctr[l])
            base[:,0] = 1
            v_angle = torch.cross(F.pad(ctr[l], (1, 0)),F.pad(base, (1,0)))[:,2].asin()
            return cat([v_len, v_angle], dim=1)

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_target["reg"]["targets"]
        disp_targets = training_target["disp"]["targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
            disp_targets[l] = disp_targets[l] / float(self.strides[l])

            # change ctr vector to angle and length 
            #disp_targets[l] = get_angle(disp_targets[l], float(self.strides[l]))

        return training_target

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask
    
    def per_target_cls_threshold_region(self, pred_cls_logits, target_dict, cls_in_boxes):
        cls_weight = None
        score_per_target = pred_cls_logits[:,target_dict['gt_classes']].sigmoid()
        in_box_logits = score_per_target.squeeze(1)[cls_in_boxes]

        if len(in_box_logits):
            score_mean = in_box_logits.mean()
            score_std = in_box_logits.std()
        else:
            score_mean = 0.0
            score_std = 0.0

        cls_pos_per_target = (score_per_target > score_mean + score_std * self.pos_sample_rate).squeeze(1)

        if self.instance_weight:
            cls_dump = pred_cls_logits[cls_in_boxes].sigmoid()
            cls_iw_recall = (cls_dump[:,target_dict["gt_classes"]] > self.pre_nms_thresh_test).sum() / ((cls_dump > self.pre_nms_thresh_test).sum() + 1e-6)
            cls_weight = 1 - cls_iw_recall

        return cls_pos_per_target,  cls_weight


    def per_target_reg_threshold_region(self, pred_box, target_dict, reg_in_boxes):
        reg_weight = None
        iou_per_target = pairwise_iou(target_dict["gt_boxes"], Boxes(pred_box))
        in_box_iou = iou_per_target.squeeze(0)[reg_in_boxes]

        if len(in_box_iou):
            iou_mean = in_box_iou.mean()
            iou_std = in_box_iou.std()
        else:
            iou_mean = 0.0
            iou_std = 0.0
        
        reg_pos_per_target = (iou_per_target >= iou_mean + iou_std * self.pos_sample_rate).squeeze(0)

        if self.instance_weight:
            iou_dump = iou_per_target.squeeze(0)[reg_in_boxes]
            iou_iw_recall = ((iou_dump > 0.5).sum() / (len(iou_dump) + 1e-6))
            reg_weight = 1 - iou_iw_recall

        return reg_pos_per_target, reg_weight


    @torch.no_grad()
    def get_threshold_region(
        self, is_in_boxes, pred_result, num_loc_list, targets_per_im, locations, image_size
    ):
        reg_in_boxes = is_in_boxes.detach()
        cls_in_boxes = is_in_boxes.detach()
        
        # prepare prediction per grid

        pred_box = cat([locations - pred_result["pred_reg"][:,:2] , locations + pred_result["pred_reg"][:,2:]], dim=1)
        pred_box[:,[0,2]] = pred_box[:,[0,2]].clamp(0, image_size[1])
        pred_box[:,[1,3]] = pred_box[:,[1,3]].clamp(0, image_size[0])

        cls_weight = []
        reg_weight = []

        for l in range(len(targets_per_im)):
            target = targets_per_im[l]
            target_dict = target.get_fields()

            # calculate logit, bbox regression threshold per object

            cls_pos_per_target, cls_weight_per_target = self.per_target_cls_threshold_region(pred_result["pred_cls"], target_dict, cls_in_boxes[:,l])
            cls_in_boxes[:,l] *= cls_pos_per_target

            reg_pos_per_target, reg_weight_per_target = self.per_target_reg_threshold_region(pred_box, target_dict, reg_in_boxes[:,l])
            reg_in_boxes[:,l] *= reg_pos_per_target

            if self.instance_weight:
                cls_weight.append(cls_weight_per_target)
                reg_weight.append(reg_weight_per_target)

        if self.instance_weight:
            cls_weight = torch.stack(cls_weight)
            reg_weight = torch.stack(reg_weight)

            assert (cls_weight >= 0).all() and (cls_weight <= 1).all()
            assert (reg_weight >= 0).all() and (reg_weight <= 1).all()

        in_boxes = {
            "cls": cls_in_boxes,
            "reg": reg_in_boxes,
            "disp": is_in_boxes,
        }
        weight = {
            "cls": cls_weight,
            "reg": reg_weight
        }

        return in_boxes, weight
    
    def no_gt_cls_target(self, locations):
        target = {
            "cls_targets": locations.new_zeros(locations.size(0)) + self.num_classes,
            "pos_inds": locations.new_zeros(locations.size(0))
        }
        return target

    def no_gt_reg_target(self, locations):
        target = {
            "reg_targets": locations.new_zeros(locations.size(0), 4),
            "pos_inds": locations.new_zeros(locations.size(0))
        }
        return target
    
    def no_gt_disp_target(self, locations):
        target = {
            "disp_targets": locations.new_zeros(locations.size(0), 2),
            "pos_inds": locations.new_zeros(locations.size(0))
        }
        return target

    def compute_DCR_for_locations(self, locations, targets, size_ranges, num_loc_list, pred_result, image_size):

        cls_training_target = []
        reg_training_target = []
        disp_training_target = []

        xs, ys = locations[:, 0], locations[:, 1]
        num_target = 0

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            image_size_per_im = image_size[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            cls_targets_per_im = targets_per_im.gt_classes
            disp_targets_per_im = targets_per_im.gt_boxes.get_centers()

            # no gt
            if bboxes.numel() == 0:
                cls_training_target.append(self.no_gt_cls_target(locations))
                reg_training_target.append(self.no_gt_reg_target(locations))
                disp_training_target.append(self.no_gt_disp_target(locations))
                continue

            # compute area and limit the regression positive to in box
            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes_pure = (reg_targets_per_im.min(dim=2)[0] > 0)

            # limit the regression range for each location
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = \
                (max_reg_targets_per_im[-len(size_ranges):] >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im[-len(size_ranges):] <= size_ranges[:, [1]])

            pred_per_im = {
                "pred_cls": cat([x[im_i].permute(1,2,0).reshape(-1, self.num_classes) for x in pred_result["pred_cls"]]),
                "pred_reg": cat([(x[im_i] * s).permute(1,2,0).reshape(-1,4) for x, s in zip(pred_result["pred_reg"], self.strides)]),
            }

            if self.is_in_boxes == "center_sampling":
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            elif self.is_in_boxes == "above_threshold":

                is_in_boxes, weights  = self.get_threshold_region(
                    is_in_boxes_pure * is_cared_in_the_level, 
                    pred_per_im,
                    num_loc_list, 
                    targets_per_im, 
                    locations,
                    image_size_per_im
                )

            else:
                is_in_boxes = is_in_boxes_pure

            def choose_among_multiple_gt(gt_area, is_in_boxes, is_cared_in_the_level):
                gt_area[is_in_boxes == 0] = INF
                gt_area[is_cared_in_the_level == 0] = INF
                return gt_area.min(dim=1)
            
            def compute_pos_inds(locations_to_gt_inds, locations_to_gt_area, num_target):
                pos_inds_per_im = locations_to_gt_inds + num_target
                pos_inds_per_im[locations_to_gt_area == INF] = -1
                return pos_inds_per_im

            # if there are still more than one objects for a location,
            # we choose the one with minimal area

            locations_to_gt_area = area[None].repeat(len(locations), 1)

            cls_locations_to_min_area, cls_locations_to_gt_inds = choose_among_multiple_gt(locations_to_gt_area, is_in_boxes["cls"], is_cared_in_the_level)
            reg_locations_to_min_area, reg_locations_to_gt_inds = choose_among_multiple_gt(locations_to_gt_area, is_in_boxes["reg"], is_cared_in_the_level)
            reg_neg_to_min_area, reg_neg_to_gt_inds = choose_among_multiple_gt(locations_to_gt_area, is_in_boxes_pure, is_cared_in_the_level)
            disp_locations_to_min_area, disp_locations_to_gt_inds = choose_among_multiple_gt(locations_to_gt_area, is_in_boxes["disp"], is_cared_in_the_level)

            # compute pos inds
            cls_pos_inds_per_im = compute_pos_inds(cls_locations_to_gt_inds, cls_locations_to_min_area, num_target)
            reg_pos_inds_per_im = compute_pos_inds(reg_locations_to_gt_inds, reg_locations_to_min_area, num_target)
            reg_neg_inds_per_im = compute_pos_inds(reg_neg_to_gt_inds, reg_neg_to_min_area, num_target)
            reg_neg_inds_per_im[reg_pos_inds_per_im != -1] = -1
            disp_pos_inds_per_im = compute_pos_inds(disp_locations_to_gt_inds, disp_locations_to_min_area, num_target)

            def compute_iou(target, pred, locations, image_size):
                pred_box = cat([locations - pred[:,:2] , locations + pred[:,2:]], dim=1)
                pred_box[:,[0,2]] = pred_box[:,[0,2]].clamp(0,image_size[1])
                pred_box[:,[1,3]] = pred_box[:,[1,3]].clamp(0, image_size[0])

                target_box = cat([locations - target[:,:2], locations + target[:,2:]], dim=1)
                target_box[:,[0,2]] = target_box[:,[0,2]].clamp(0, image_size[1])
                target_box[:,[1,3]] = target_box[:,[1,3]].clamp(0, image_size[0])

                return matched_boxlist_iou(Boxes(target_box), Boxes(pred_box))
            # compute target
            cls_targets_per_im = cls_targets_per_im[cls_locations_to_gt_inds]
            cls_targets_per_im[cls_locations_to_min_area == INF] = self.num_classes
            reg_targets_per_im = reg_targets_per_im[torch.arange(len(reg_locations_to_gt_inds)),reg_locations_to_gt_inds,:]
            iou_targets_per_im = compute_iou(reg_targets_per_im, pred_per_im["pred_reg"], locations, image_size_per_im)
            disp_targets_per_im = locations - disp_targets_per_im[disp_locations_to_gt_inds]
            disp_targets_per_im[disp_locations_to_min_area == INF] = 0


            reg_label_per_im = -torch.ones_like(reg_pos_inds_per_im, device=reg_pos_inds_per_im.device).long()
            reg_label_per_im[reg_pos_inds_per_im] = targets_per_im.gt_classes[reg_locations_to_gt_inds[reg_pos_inds_per_im]]

            num_target += len(targets_per_im)

            cls_training_target_per_im = {
                "targets": cls_targets_per_im,
                "pos_inds" : cls_pos_inds_per_im,
            }
            reg_training_target_per_im = {
                "targets": reg_targets_per_im,
                "pos_inds": reg_pos_inds_per_im,
                "neg_inds": reg_neg_inds_per_im,
                "ious": iou_targets_per_im,
                "labels": reg_label_per_im,
            }
            disp_training_target_per_im = {
                "targets": disp_targets_per_im,
                "pos_inds": disp_pos_inds_per_im,
            }

            if self.instance_weight:
                cls_weight_per_im = torch.ones(len(cls_targets_per_im), dtype=torch.float32, device=weights["cls"].device)
                cls_weight_per_im[cls_pos_inds_per_im] += weights["cls"][cls_locations_to_gt_inds][cls_pos_inds_per_im]
                cls_training_target_per_im["weights"] = cls_weight_per_im

                reg_weight_per_im = torch.ones(len(reg_targets_per_im), dtype=torch.float32, device=weights["reg"].device)
                reg_weight_per_im[reg_pos_inds_per_im] += weights["reg"][reg_locations_to_gt_inds][reg_pos_inds_per_im]
                reg_training_target_per_im["weights"] = reg_weight_per_im
            
            cls_training_target.append(cls_training_target_per_im)
            reg_training_target.append(reg_training_target_per_im)
            disp_training_target.append(disp_training_target_per_im)

        training_target = {
            "cls": defaultdict(list, {k:[] for k in list(cls_training_target[0].keys())}),
            "reg": defaultdict(list, {k:[] for k in list(reg_training_target[0].keys())}),
            "disp": defaultdict(list, {k:[] for k in list(disp_training_target[0].keys())}),
        }

        for kind, training_target_list in zip(list(training_target.keys()),
            [cls_training_target, reg_training_target, disp_training_target]):
            for training_target_per_kind in training_target_list:
                for k, v in training_target_per_kind.items():
                    training_target[kind][k].append(v)

        return training_target

    def generate_cls_instances(self, training_target, pred_result):
        # Create instance
        cls_instances = Instances((0, 0))

        # push training target to instance
        cls_instances.cls_targets = cat([
            x.reshape(-1) for x in training_target["targets"]
        ])
        cls_instances.pos_inds = cat([
            x.reshape(-1) for x in training_target["pos_inds"]
        ])

        # push pred result to instance
        cls_instances.pred_cls = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in pred_result["pred_cls"] 
        ], dim=0,)

        # push weight if instance weight is True
        if self.instance_weight:
            cls_instances.weight = cat([
                x.reshape(-1) for x in training_target["weight"]
            ])

        return cls_instances

    def generate_reg_instances(self, training_target, pred_result):
        # Create instance
        reg_instances = Instances((0, 0))

        # push training target to instance
        reg_instances.reg_targets = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1, 4) for x in training_target["targets"]
        ], dim=0)
        reg_instances.iou_targets = cat([
            x.reshape(-1) for x in training_target["ious"]
        ], dim=0)
        reg_instances.pos_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_target["pos_inds"]
        ])
        reg_instances.neg_inds = cat([
            x.reshape(-1) for x in training_target["neg_inds"]
        ])

        # push pred result to instance
        reg_instances.pred_reg = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in pred_result["pred_reg"]
        ], dim=0,)
        reg_instances.pred_iou = cat([
            x.permute(0, 2, 3, 1).reshape(-1) for x in pred_result["pred_iou"]
        ], dim=0,)

        # push weight if instance weight is True
        if self.instance_weight:
            reg_instances.reg_weight = cat([
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in training_target["weight"]
            ])

        return reg_instances

    def generate_disp_instances(self, training_target, pred_result):
        # Create instance
        disp_instances = Instances((0, 0))

        # push training target to instance
        disp_instances.disp_targets = cat([
            x.reshape(-1, 2) for x in training_target["targets"]
        ])
        disp_instances.pos_inds = cat([
            x.reshape(-1) for x in training_target["pos_inds"]
        ])

        # push pred result to instnace
        disp_instances.pred_disp = cat([
            x.permute(0, 2, 3, 1).reshape(-1, 2) for x in pred_result["pred_disp"]
        ], dim=0)

        # push weight if instance weight is True
        if self.instance_weight:
            disp_instances.weight = cat([
                x.reshape(-1) for x in training_target["weight"]
            ])

        return disp_instances

    def losses(self, pred_result, locations, gt_instances, image_size):

        # get ground truth training target
        training_target = self._get_ground_truth(locations, gt_instances, pred_result, image_size)

        # generate fast axis instance with pred result and training target
        cls_instances = self.generate_cls_instances(training_target["cls"], pred_result)
        reg_instances = self.generate_reg_instances(training_target["reg"], pred_result)
        disp_instances = self.generate_disp_instances(training_target["disp"], pred_result)

        # compute loss
        return self.dcr_losses({
            "cls": cls_instances, 
            "reg": reg_instances, 
            "disp": disp_instances,
            })
    
    def compute_recall_weight(self, pred_pos, target_pos, gt_channel=0):

        tp = (pred_pos * target_pos)
        fn = (~pred_pos * target_pos)

        return tp.sum() / (tp.sum() + fn.sum())

    def dcr_cls_losses(self, cls_instances, num_gpus):
        cls_targets = cls_instances.cls_targets.flatten()

        #cls_pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        cls_pos_inds = cls_instances.pos_inds != -1
        num_cls_pos_local = cls_pos_inds.sum()
        total_cls_num_pos = reduce_sum(num_cls_pos_local).item()
        num_cls_pos_avg = max(total_cls_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(cls_instances.pred_cls)
        class_target[cls_pos_inds, cls_targets[cls_pos_inds]] = 1

        if self.instance_weight:
            # compute tp, fp, fn weight
            tp = (cls_instances.pred_cls.sigmoid() > self.pre_nms_thresh_test) * class_target
            fp = (cls_instances.pred_cls.sigmoid() > self.pre_nms_thresh_test) * (1 - class_target).bool()
            cls_instances.weight[fp.any(dim=1)] += (1 - tp.sum() / (tp.sum() + fp.sum() + 1e-6))
            cls_instances.weight /= cls_instances.weight.mean()

        class_loss = sigmoid_focal_loss_jit(
            cls_instances.pred_cls,
            class_target,
            weight=cls_instances.weight.unsqueeze(1) if self.instance_weight else None,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_cls_pos_avg

        loss = {
            "loss_dcr_cls": class_loss,
        }

        return loss

    def dcr_reg_losses(self, reg_instances, num_gpus):
        reg_pos_inds = reg_instances.pos_inds != -1
        num_reg_pos_local = reg_pos_inds.sum()
        total_reg_num_pos = reduce_sum(num_reg_pos_local).item()
        num_reg_pos_avg = max(total_reg_num_pos / num_gpus, 1.0)

        reg_neg_inds = reg_instances.neg_inds != -1
        num_reg_neg_local = reg_neg_inds.sum()
        total_reg_num_neg = reduce_sum(num_reg_neg_local).item()
        num_reg_neg_avg= max(total_reg_num_neg / num_gpus, 1.0)

        if self.instance_weight:
            # compute tp, fp, fn weight
            tp = (reg_instances.box_logits.sigmoid() > self.pre_nms_thresh_test) * target_anchor
            fp = (reg_instances.box_logits.sigmoid() > self.pre_nms_thresh_test) * (1 - target_anchor).bool()
            reg_instances.target_weight[fp] += (1 - tp.sum() / (tp.sum() + fp.sum() + 1e-6))
            reg_instances.target_weight /= reg_instances.target_weight.mean()
            reg_instances.reg_weight /= reg_instances.reg_weight.mean()

        if reg_pos_inds.numel() + reg_neg_inds.numel() > 0:
            iou_loss = sigmoid_focal_loss_jit(
                reg_instances.pred_iou[reg_pos_inds + reg_neg_inds],
                reg_instances.iou_targets[reg_pos_inds + reg_neg_inds],
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum"
            ) / (num_reg_pos_avg + num_reg_neg_avg)
        else:
            iou_loss = reg_instances.pred_iou.sum() * 0

        if reg_pos_inds.numel() > 0:

            reg_loss = self.loc_loss_func(
                reg_instances.pred_reg[reg_pos_inds],
                reg_instances.reg_targets[reg_pos_inds],
                #weight=reg_instances.reg_weight.unsqueeze(1) if hasattr(reg_instances, "reg_weight") else None,
            ) / num_reg_pos_avg

        else:
            reg_loss = reg_instances.pred_reg.sum() * 0
        
        assert((1 - reg_loss.item()) > 0)

        loss = {
            "loss_dcr_reg": reg_loss,
            "loss_dcr_iou": iou_loss * (1 - reg_loss.detach()),
        }

        return loss

    def dcr_disp_losses(self, disp_instances, num_gpus):
        disp_pos_inds = disp_instances.pos_inds != -1
        num_disp_pos_local = disp_pos_inds.sum()
        total_disp_num_pos = reduce_sum(num_disp_pos_local).item()
        num_disp_pos_avg = max(total_disp_num_pos / num_gpus, 1.0)

        disp_pos_loss = self.disp_loss_func(
            disp_instances.pred_disp[disp_pos_inds],
            disp_instances.disp_targets[disp_pos_inds],
        ) / num_disp_pos_avg

        disp_neg_loss = self.disp_loss_func(
            disp_instances.pred_disp[~disp_pos_inds],
            disp_instances.disp_targets[~disp_pos_inds]
        ) / (len(disp_pos_inds) - num_disp_pos_avg)

        loss = {
            "loss_dcr_disp_pos": disp_pos_loss,
            "loss_dcr_disp_neg": disp_neg_loss
        }

        return loss

    def dcr_losses(self, instances):

        num_classes = instances["cls"].pred_cls.size(1)
        assert num_classes == self.num_classes
        num_gpus = get_world_size()
        losses = {}

        losses.update(self.dcr_cls_losses(instances["cls"], num_gpus))
        losses.update(self.dcr_reg_losses(instances["reg"], num_gpus))
        losses.update(self.dcr_disp_losses(instances["disp"], num_gpus))

        return instances, losses

    def predict_proposals(
            self, logits_pred, reg_pred, box_logits,
            locations, image_sizes, training_target=None
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []
        cls_analysis = []
        reg_analysis = []
        pt_analysis = []

        bundle = {
            "l": locations,
            "r": reg_pred, "t": box_logits,
            "s": self.strides, 
        }

        if training_target is not None:
            reg_target = training_target["reg"]["reg_targets"]
            pt_target = training_target["reg"]["pos_inds"]
            reg_label = training_target["reg"]["reg_label"]

            bundle["rt"] = reg_target
            bundle["pt"] = pt_target
            bundle["rl"] = reg_label

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = logits_pred[0]
            r = per_bundle["r"] * per_bundle["s"]
            t = per_bundle["t"] 
            targets = None

            if training_target is not None:
                rt = per_bundle["rt"] * per_bundle["s"]
                pt = per_bundle["pt"]
                rl = per_bundle["rl"]
                targets = [training_target["cls"], rt, pt, rl] if i == 0  else [None, rt, pt, rl]

            boxes, analysis = self.forward_for_single_feature_map(
                l, o, r, t, image_sizes, targets
            )
            sampled_boxes.append(boxes)
            if "cls" in analysis:
                cls_analysis.append(analysis["cls"])
            if "reg" in analysis:
                reg_analysis.append(analysis["reg"])
            if "pt" in analysis:
                pt_analysis.append(analysis["pt"])

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists, [cls_analysis, reg_analysis, pt_analysis]

    def calc_stat(self, true, positive):
        tp = true * positive
        fn = ~true * positive
        fp = true * ~positive

        jid = tp.sum() / (tp.sum() + fn.sum() + fp.sum())

        return tp.sum(), fn.sum(), fp.sum(), jid

    def forward_for_single_feature_map(
            self, locations, logits_pred, reg_pred,
            box_logits, image_sizes, target_list=None
    ):

        results = []
        analysis = defaultdict(list)

        N, C, Hc, Wc = logits_pred.shape

        # put in the same format as locations

        cls_map = logits_pred
        logits_pred = logits_pred.view(N, C, Hc, Wc).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()

        _, _, H, W = reg_pred.shape

        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        box_logits = box_logits.view(N, 1, H, W).permute(0, 2, 3, 1)
        box_logits = box_logits.reshape(N, -1).sigmoid()

        # calculate true positive, false positive, false negative of prediction
        if target_list is not None:
            cls_trg, reg_trg, pt_trg, reg_label = target_list
            if cls_trg is not None:
                true_label = cls_trg["labels"][0]
                label_pos = cls_trg["pos_inds"][0]

                true = torch.zeros_like(logits_pred).bool()
                true[:,label_pos,true_label[label_pos]] = 1
                analysis["cls"].append(self.calc_stat(true, logits_pred > self.pre_nms_thresh))

            reg_trg = cat([locations - reg_trg[:,:2], locations + reg_trg[:,2:]],dim=1)
            box_pred = cat([locations - box_regression[0,:,:2], locations + box_regression[0,:,2:]], dim=1)
            reg_trg[reg_trg < 0] = 0
            box_pred[box_pred < 0] = 0
            iou_trg = matched_boxlist_iou(Boxes(reg_trg), Boxes(box_pred))
            iou_trg[~pt_trg] = 0

            #analysis["reg"].append([self.calc_stat(iou_trg > 0.5, box_logits[0] > self.pre_nms_thresh)])
            analysis["pt"].append(self.calc_stat(pt_trg, box_logits[0] > self.pre_nms_thresh))
            #iou_trg[pt_trg]

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.

        if target_list is not None:
            candidate_inds = iou_trg.unsqueeze(0) > self.pre_nms_thresh
            #candidate_inds = box_logits > self.pre_nms_thresh
        else:
            candidate_inds = box_logits > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        """
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)
        """

        for i in range(N):
            per_cls_map = cls_map.sigmoid()
            per_box_regression = box_regression[i]
            per_box_box_logits = box_logits[i]
            per_candidate_inds = candidate_inds[i]
            per_pre_nms_top_n = pre_nms_top_n[i]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_box_regression = per_box_regression[per_box_loc]
            per_box_box_logits = per_box_box_logits[per_box_loc]
            per_locations = locations[per_box_loc]

            if target_list is not None:
                per_reg_label = reg_label
                per_reg_label = per_reg_label[per_box_loc]
                per_iou_trg = iou_trg[per_box_loc]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_box_logits, top_k_indices = \
                    per_box_box_logits.topk(per_pre_nms_top_n, sorted=False)
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                if target_list is not None:
                    per_iou_trg = per_iou_trg[top_k_indices]
                    per_reg_label = per_reg_label[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            """
            per_box_cls = per_box_cls[per_candidate_inds]
            per_class = per_candidate_nonzeros[:, 1]
            per_class = per_class[top_k_indices]
            """

            if target_list is not None and False:
                detections = detections[per_reg_label != -1]
                per_iou_trg = per_iou_trg[per_reg_label != -1]
                per_box_cls = per_box_box_logits[per_reg_label != -1]
                per_class = per_reg_label[per_reg_label != -1]
                per_locations = per_locations[per_reg_label != -1]
                per_reg_label = per_reg_label[per_reg_label != -1]
            else:
                from torchvision.ops import roi_align
                roi = cat([torch.zeros((len(detections),1), device=detections.device),detections/4],dim=1)
                roi[:,[0,2]] = roi[:,[0,2]].clamp(0, Wc)
                roi[:,[1,3]] = roi[:,[1,3]].clamp(0, Hc)
                per_box_cls = roi_align(per_cls_map, roi, output_size=1)

                per_box_cls = per_box_cls.reshape(-1, self.num_classes)
                per_candidate_inds = (per_box_cls > self.pre_nms_thresh)

                if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                    cls_threshold = per_box_cls.flatten().topk(per_pre_nms_top_n,sorted=False)[0].min()
                    per_candidate_inds = per_box_cls > cls_threshold

                per_candidate_nonzeros = per_candidate_inds.nonzero()

                per_box_cls = per_box_cls[per_candidate_inds]
                per_box_loc = per_candidate_nonzeros[:,0]
                per_class = per_candidate_nonzeros[:,1]
                detections = detections[per_box_loc]
                per_box_box_logits = per_box_box_logits[per_box_loc]
                per_locations = per_locations[per_box_loc]
                
            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = per_box_cls
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            results.append(boxlist)

        return results, analysis

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

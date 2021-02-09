from collections import defaultdict
import logging
from detectron2.structures.boxes import matched_boxlist_iou
import torch
from torch import nn
import torch.nn.functional as F
import copy

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit
import sklearn.mixture as skm

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms, IOUExtendLoss
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
        self.loc_loss_func = IOUExtendLoss(cfg.MODEL.DCR.LOC_LOSS_TYPE)
        self.iou_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.disp_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

        self.pre_nms_thresh_test = cfg.MODEL.DCR.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.DCR.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.DCR.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.DCR.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.DCR.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.DCR.NUM_CLASSES
        self.strides = cfg.MODEL.DCR.FPN_STRIDES
        self.instance_weight = cfg.MODEL.DCR.INSTANCE_WEIGHT
        self.vis_period = cfg.VIS_PERIOD
        self.reg_on_cls = cfg.MODEL.DCR.REG_ON_CLS


        # generate sizes of interest
        soi = []
        prev_size = 0
        for s in cfg.MODEL.DCR.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
    
    def draw_field(self, location, pred_disp, self_dir, pos_inds, image, i):
        import matplotlib.pyplot as plt
        N, _, H, W = pred_disp.shape
        plt.figure(figsize=(W*2//10, H*2//10))
        pred_disp = pred_disp.permute(0,2,3,1)
        location = location.reshape(N, H, W, -1)
        X = location[0,:,:,0].detach().cpu().numpy()
        Y = location[0,:,:,1].detach().cpu().numpy()
        U = pred_disp[0,:,:,0].detach().cpu().numpy()
        V = pred_disp[0,:,:,1].detach().cpu().numpy()
        self_dir = self_dir[0].detach().cpu().numpy()
        pos_inds = pos_inds[0].detach().cpu().numpy()
        image -= image.min()
        image /= image.max()
        image = F.interpolate(image.unsqueeze(0), size=(H, W)).squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,::-1]
        import numpy as np
        plt.title("scales with x view")
        M = np.hypot(V, U)
        Q = plt.quiver(X[~self_dir], Y[~self_dir], U[~self_dir], -V[~self_dir],  M[~self_dir], units='x', width=0.22,
                    scale=1 / 1)
        qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                        coordinates='figure')
        plt.scatter(X, Y, color='k', s=0.01)
        plt.scatter(X[pos_inds], Y[pos_inds], color='blue', s=40)
        plt.scatter(X[self_dir * pos_inds], Y[self_dir * pos_inds], color='red', s=20)

        plt.imshow(image)
        plt.tight_layout()


        plt.savefig("output/dcrv2/server-14/inference/field_{}_{}.png".format(H*W,i))

    def write_debug_img(self, num_loc_list, locations, reg_pos_inds_per_im, cls_pos_inds_per_im, disp_targets_per_im, disp_pos_inds_per_im, centers, bboxes):
        storage = get_event_storage()
        st = 0
        for l, num_loc in enumerate(num_loc_list):
            en = st + num_loc
            X = locations[st:en,0].detach().cpu().numpy()
            Y = locations[st:en,1].detach().cpu().numpy()
            U = disp_targets_per_im[st:en,0].detach().cpu().numpy()
            V = disp_targets_per_im[st:en,1].detach().cpu().numpy()
            reg_pos = (reg_pos_inds_per_im[st:en] != -1).detach().cpu().numpy()
            cls_pos = (cls_pos_inds_per_im[st:en] != -1).detach().cpu().numpy()
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            import cv2
            M = disp_pos_inds_per_im[st:en].detach().cpu().numpy()
            fig = plt.figure(figsize=(X.max()/60, Y.max()/60))
            plt.quiver(X, Y, U, V, M, units="xy", angles="xy", width=1, scale=1)
            plt.scatter(X[reg_pos], Y[reg_pos], color='red', s=40, alpha=0.7, marker='+')
            plt.scatter(X[cls_pos], Y[cls_pos], color='blue', s=40, alpha=0.7, marker='x')
            plt.scatter(centers[:,0].detach().cpu().numpy(), centers[:,1].detach().cpu(), c=np.arange(len(bboxes)), cmap=plt.cm.rainbow, s=40)
            buf = io.BytesIO()
            ax = plt.gca()
            for rl ,rec in enumerate(bboxes):
                import matplotlib.patches as patches
                rect = patches.Rectangle(
                    (rec[0].item(), rec[1].item()),
                    rec[2] - rec[0],
                    rec[3] - rec[1],
                    linewidth=2,
                    fill= False
                )
                ax.add_patch(rect)
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_name = 'lvl{}'.format(l)
            storage.put_image(vis_name, torch.tensor(img).permute(2,0,1))
            st = en
            plt.close()

    def iterate_disp(self, pred_disp, pos_inds = None, self_dir = None, image = None):

        #num_it = int((max(self.pos_sample_rate + 0.5, 0)) // 0.2)
        num_it = 4
        num_touch = [x.new_ones(x.shape[0],x.shape[2], x.shape[3]).long() for x in pred_disp]
        result_cent = [x.new_ones(x.shape[0], 3,x.shape[2], x.shape[3]).long() for x in pred_disp]

        for i in range(num_it):
            for l, (disp_per_level, num_touch_per_level) in enumerate(zip(pred_disp, num_touch)):
                N, _, H, W  = disp_per_level.shape
                location = disp_per_level.new_ones(H, W).nonzero().reshape(1,H, W, 2)[:,:,:,[1,0]]
                #location = cat([disp_per_level.new_ones(H, W).nonzero().reshape(1, H, W, 2)] * N).permute(0,3,1,2)
                #self.draw_field(location, disp_per_level, self_dir, pos_inds!=-1, image, i)
                location = location.permute(0,3,1,2)
                pred_cent = (location + disp_per_level).long()
                # Need clamp here
                pred_cent[:,0,:,:] = pred_cent[:,0].clamp(0,W-1)
                pred_cent[:,1,:,:] = pred_cent[:,1].clamp(0,H-1)
                pred_cent = cat([torch.arange(N, device=pred_cent.device).reshape(-1,1).repeat(1,H*W).reshape(N,1,H,W), pred_cent], dim=1)

                pred_cent_flatten = pred_cent.permute(0,2,3,1).reshape(-1,3)
                assert torch.logical_and(pred_cent_flatten[:,1].max() < W, pred_cent_flatten[:,1].min() >= 0).item()
                assert torch.logical_and(pred_cent_flatten[:,2].max() < H, pred_cent_flatten[:,2].min() >= 0).item()

                unique_pred_cent, pred_count = pred_cent_flatten.unique(dim=0, return_counts=True)
                new_touch = torch.zeros_like(num_touch_per_level)
                new_touch[unique_pred_cent[:,0],unique_pred_cent[:,2], unique_pred_cent[:,1]] = pred_count

                """
                import matplotlib.pyplot as plt
                plt.figure()
                mask = (new_touch[new_touch!=0]!=1).reshape(-1)
                Y = new_touch.nonzero()[mask,1]
                X = new_touch.nonzero()[mask,2]
                sc = plt.scatter(X.detach().cpu().numpy(), H - 1 - Y.detach().cpu().numpy(), 10 ,c=new_touch[0,Y,X].detach().cpu().numpy())
                plt.colorbar(sc)
                plt.xlim([0,W])
                plt.ylim([0,H])
                plt.savefig('output/dcrv2/server-14/inference/num_touch_{}_{}.png'.format(H*W,i))
                """

                inner_disp = disp_per_level[pred_cent_flatten[:,0], :, pred_cent_flatten[:,2], pred_cent_flatten[:,1]].reshape(N, H, W, 2).permute(0,3,1,2)
                pred_disp[l] = inner_disp + disp_per_level
                num_touch[l] = new_touch
                result_cent[l] = pred_cent

        return pred_disp, num_touch, result_cent

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
        loc_to_stride = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )
            loc_to_stride_per_level = loc_per_level.new_tensor(self.strides[l])
            loc_to_stride.append(
                loc_to_stride_per_level[None].expand(num_loc_list[l],-1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        loc_to_stride = torch.cat(loc_to_stride,dim=0)
        locations = torch.cat(locations, dim=0)

        training_target = self.compute_DCR_for_locations(
            locations, gt_instances, loc_to_size_range, loc_to_stride, num_loc_list, pred_result, image_size
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
            disp_targets[l] = torch.round(disp_targets[l])
            disp_targets[l] = disp_targets[l].clamp(-1,1)

            #  convert disp target from vector to logit
            #  [l, r, t, b]
            disp_targets[l] = torch.stack([(disp_targets[l][:,0] == -1), \
                (disp_targets[l][:,0] == 1), \
                (disp_targets[l][:,1] == -1), \
                (disp_targets[l][:,1] == 1)], dim=1)

            """
            X = locations[:15200,0].detach().cpu().numpy()
            Y = locations[:15200,1].detach().cpu().numpy()
            U = disp_targets[l][:,0].detach().cpu().numpy()
            V = disp_targets[l][:,1].detach().cpu().numpy()
            import matplotlib.pyplot as plt
            import numpy as np
            M = np.arctan2(U, V)
            plt.figure(figsize=(15,10))
            plt.quiver(X, Y, U , V, M, width=0.0022, scale=1/0.015)
            plt.savefig('output/dcrv2/server-14/inference/clamp.png')
            """
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

        if len(in_box_logits) > 1:
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

        return cls_pos_per_target, cls_weight


    def per_target_reg_threshold_region(self, pred_box, target_dict, reg_in_boxes):
        reg_weight = None
        iou_per_target = pairwise_iou(target_dict["gt_boxes"], Boxes(pred_box))
        in_box_iou = iou_per_target.squeeze(0)[reg_in_boxes]

        if len(in_box_iou) > 1:
            iou_mean = in_box_iou.mean()
            iou_std = in_box_iou.std()
        else:
            iou_mean = 0.0
            iou_std = 0.0
        
        #print(in_box_iou.max())
        reg_pos_per_target = (iou_per_target >= iou_mean + iou_std * self.pos_sample_rate).squeeze(0)
        reg_neg_per_target = (iou_per_target < iou_mean + iou_std * self.pos_sample_rate).squeeze(0)

        if self.instance_weight:
            iou_dump = iou_per_target.squeeze(0)[reg_in_boxes]
            iou_iw_recall = ((iou_dump > 0.5).sum() / (len(iou_dump) + 1e-6))
            reg_weight = 1 - iou_iw_recall

        return reg_pos_per_target, reg_neg_per_target, reg_weight


    @torch.no_grad()
    def get_threshold_region(
        self, is_in_boxes, pred_result, targets_per_im, locations, image_size
    ):

        reg_in_boxes = copy.deepcopy(is_in_boxes)
        reg_neg_boxes = copy.deepcopy(is_in_boxes)
        cls_in_boxes = copy.deepcopy(is_in_boxes)
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

            reg_pos_per_target, reg_neg_per_target, reg_weight_per_target = self.per_target_reg_threshold_region(pred_box, target_dict, reg_in_boxes[:,l])
            reg_in_boxes[:,l] *= reg_pos_per_target
            reg_neg_boxes[:,l] *= reg_neg_per_target

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
            "reg_neg": reg_neg_boxes,
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

    def compute_DCR_for_locations(self, locations, targets, size_ranges, loc_to_strides, num_loc_list, pred_result, image_size):

        cls_training_target = []
        reg_training_target = []
        disp_training_target = []

        xs, ys = locations[:, 0], locations[:, 1]
        num_target = 0

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            image_size_per_im = image_size[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            centers = targets_per_im.gt_boxes.get_centers()
            cls_targets_per_im = targets_per_im.gt_classes
            disp_targets_per_im = targets_per_im.gt_boxes.get_centers()

            # no gt
            if bboxes.numel() == 0:
                cls_training_target.append(self.no_gt_cls_target(locations))
                reg_training_target.append(self.no_gt_reg_target(locations))
                disp_training_target.append(self.no_gt_disp_target(locations))
                continue

            dx = centers[:, 0][None] - xs[:, None] 
            dy = centers[:, 1][None] - ys[:, None]
            dw = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(0).repeat(len(dx),1)
            dh = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(0).repeat(len(dx),1)
            area_per_im = (dw * dh).sqrt()
            reg_targets_per_im = torch.stack([dx, dy, dw, dh], dim=2)
            dist_ratio = ((dx ** 2 + dy ** 2)).sqrt()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            in_box_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes_pure = in_box_per_im.min(dim=2)[0] > 0
            is_in_boxes_surround = (in_box_per_im.min(dim=2)[0] + loc_to_strides > 0)
            no_in_box_pure = is_in_boxes_pure.sum(dim=0) <= 10
            if no_in_box_pure.any().item():
                is_in_boxes_pure[:,no_in_box_pure] = is_in_boxes_surround[:,no_in_box_pure]

            is_cared_in_the_level = (area_per_im >= size_ranges[:,0][:,None]) * \
                 (area_per_im < size_ranges[:,1][:,None])

            pred_per_im = {
                "pred_cls": cat([x[im_i].permute(1,2,0).reshape(-1, self.num_classes) for x in pred_result["pred_cls"]]),
                "pred_reg": cat([(x[im_i] * s).permute(1,2,0).reshape(-1,4) for x, s in zip(pred_result["pred_reg"], self.strides)]),
            }

            is_in_boxes, weights  = self.get_threshold_region(
                is_in_boxes_pure * is_cared_in_the_level, 
                pred_per_im,
                targets_per_im, 
                locations,
                image_size_per_im
            )
            
            def choose_among_multiple_gt(gt_area, is_in_boxes, is_cared_in_the_level):
                gt_area = copy.deepcopy(gt_area)
                gt_area[is_in_boxes == 0] = INF
                gt_area[is_cared_in_the_level == 0] = INF
                return gt_area.min(dim=1)
            
            def compute_pos_inds(locations_to_gt_inds, locations_to_gt_area, num_target):
                pos_inds_per_im = locations_to_gt_inds + num_target
                pos_inds_per_im[locations_to_gt_area == INF] = -1
                return pos_inds_per_im

            # if there are still more than one objects for a location,
            # we choose the one with minimal area

            cls_locations_to_min_area, cls_locations_to_gt_inds = choose_among_multiple_gt(dist_ratio, is_in_boxes["cls"], is_cared_in_the_level)
            reg_locations_to_min_area, reg_locations_to_gt_inds = choose_among_multiple_gt(dist_ratio, is_in_boxes["reg"], is_cared_in_the_level)
            disp_locations_to_min_area, disp_locations_to_gt_inds = choose_among_multiple_gt(dist_ratio, is_in_boxes_pure, is_cared_in_the_level)

            cls_pos_inds_per_im = compute_pos_inds(cls_locations_to_gt_inds, cls_locations_to_min_area, num_target)
            reg_pos_inds_per_im = compute_pos_inds(reg_locations_to_gt_inds, reg_locations_to_min_area, num_target)

            if self.reg_on_cls:
                target_ind = (reg_pos_inds_per_im == -1) * (cls_pos_inds_per_im != -1)
                reg_pos_inds_per_im[target_ind] = cls_pos_inds_per_im[target_ind]

            disp_pos_inds_per_im = compute_pos_inds(disp_locations_to_gt_inds, disp_locations_to_min_area, num_target)

            reg_neg_inds_per_im = is_in_boxes["reg_neg"].sum(dim=1).bool()
            reg_neg_inds_per_im[reg_pos_inds_per_im != -1] = False

            def compute_iou(target, pred, locations, image_size):
                pred_ctr = locations + pred[:,:2]
                pred_box = cat([pred_ctr - pred[:,2:]/2 , pred_ctr + pred[:,2:]/2], dim=1)
                pred_box[:,[0,2]] = pred_box[:,[0,2]].clamp(0,image_size[1])
                pred_box[:,[1,3]] = pred_box[:,[1,3]].clamp(0, image_size[0])

                target_ctr = locations + target[:,:2]
                target_box = cat([target_ctr - target[:,2:]/2, target_ctr + target[:,2:]/2], dim=1)
                target_box[:,[0,2]] = target_box[:,[0,2]].clamp(0, image_size[1])
                target_box[:,[1,3]] = target_box[:,[1,3]].clamp(0, image_size[0])

                return matched_boxlist_iou(Boxes(target_box), Boxes(pred_box))
            # compute target
            cls_targets_per_im = cls_targets_per_im[cls_locations_to_gt_inds]
            cls_targets_per_im[cls_locations_to_min_area == INF] = self.num_classes
            reg_targets_per_im = reg_targets_per_im[torch.arange(len(reg_locations_to_gt_inds)),reg_locations_to_gt_inds,:]
            iou_targets_per_im = compute_iou(reg_targets_per_im, pred_per_im["pred_reg"], locations, image_size_per_im)
            disp_targets_per_im = disp_targets_per_im[disp_locations_to_gt_inds] - locations
            disp_targets_per_im[disp_locations_to_min_area == INF] = 0

            try:
                assert (cls_pos_inds_per_im.unique()[(cls_pos_inds_per_im.unique() != -1)] == \
                    reg_pos_inds_per_im.unique()[(reg_pos_inds_per_im.unique() != -1)]).all()
                assert len(cls_pos_inds_per_im.unique()) == (len(targets_per_im) + 1)
                assert len(cls_pos_inds_per_im.unique()) == len(disp_pos_inds_per_im.unique())
            except:
                self.write_debug_img(num_loc_list, locations, reg_pos_inds_per_im,
                    cls_pos_inds_per_im, disp_targets_per_im, disp_pos_inds_per_im, centers, bboxes)

            storage = get_event_storage()
            if self.vis_period > 0:
                if storage.iter % self.vis_period == 0 and im_i == 0:
                    self.write_debug_img(num_loc_list, locations, reg_pos_inds_per_im,
                        cls_pos_inds_per_im, disp_targets_per_im, disp_pos_inds_per_im, centers, bboxes)

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
                cls_weight_per_im[cls_pos_inds_per_im != -1] += weights["cls"][cls_locations_to_gt_inds][cls_pos_inds_per_im != -1]
                cls_training_target_per_im["weights"] = cls_weight_per_im

                reg_weight_per_im = torch.ones(len(reg_targets_per_im), dtype=torch.float32, device=weights["reg"].device)
                reg_weight_per_im[reg_pos_inds_per_im != -1] += weights["reg"][reg_locations_to_gt_inds][reg_pos_inds_per_im != -1]
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
                x.reshape(-1) for x in training_target["weights"]
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
            reg_instances.weight = cat([
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in training_target["weights"]
            ])

        return reg_instances

    def generate_disp_instances(self, training_target, pred_result):
        # Create instance
        disp_instances = Instances((0, 0))

        # push training target to instance
        disp_instances.disp_targets = cat([
            x.reshape(-1, 4) for x in training_target["targets"]
        ])
        disp_instances.pos_inds = cat([
            x.reshape(-1) for x in training_target["pos_inds"]
        ])

        # push pred result to instnace
        disp_instances.pred_disp = cat([
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in pred_result["pred_disp"]
        ], dim=0)

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

        reg_neg_inds = reg_instances.neg_inds
        num_reg_neg_local = reg_neg_inds.sum()
        total_reg_num_neg = reduce_sum(num_reg_neg_local).item()
        num_reg_neg_avg= max(total_reg_num_neg / num_gpus, 1.0)

        if self.instance_weight:
            reg_instances.weight /= reg_instances.weight.mean()

        if reg_pos_inds.numel() + reg_neg_inds.numel() > 0:
            iou_loss = self.iou_loss_func(
                reg_instances.pred_iou[reg_pos_inds + reg_neg_inds],
                reg_instances.iou_targets[reg_pos_inds + reg_neg_inds],
            ) / (num_reg_pos_avg + num_reg_neg_avg)
        else:
            iou_loss = reg_instances.pred_iou.sum() * 0

        if reg_pos_inds.numel() > 0:
            reg_ctr = reg_instances.pred_reg[reg_pos_inds][:,:2]
            reg_wh = reg_instances.pred_reg[reg_pos_inds][:,2:]
            trg_ctr = reg_instances.reg_targets[reg_pos_inds][:,:2]
            trg_wh = reg_instances.reg_targets[reg_pos_inds][:,2:]

            reg_loss = self.loc_loss_func(
                torch.cat([reg_ctr - reg_wh / 2, reg_ctr + reg_wh / 2],dim=1),
                torch.cat([trg_ctr - trg_wh / 2, trg_ctr + trg_wh / 2],dim=1),
                weight=reg_instances.reg_weight.unsqueeze(1) if hasattr(reg_instances, "reg_weight") else None,
            ) / num_reg_pos_avg

        else:
            reg_loss = reg_instances.pred_reg.sum() * 0
        
        iou_weight = max(1 - reg_loss.detach(), 0)

        loss = {
            "loss_dcr_reg": reg_loss,
            "loss_dcr_iou": iou_loss * iou_weight,
            #"loss_dcr_iou": iou_loss ,
        }

        return loss

    def dcr_disp_losses(self, disp_instances, num_gpus):
        disp_pos_inds = disp_instances.pos_inds != -1
        num_disp_pos_local = disp_pos_inds.sum()
        total_disp_num_pos = reduce_sum(num_disp_pos_local).item()
        num_disp_pos_avg = max(total_disp_num_pos / num_gpus, 1.0)

        disp_loss = sigmoid_focal_loss_jit(
            disp_instances.pred_disp,
            disp_instances.disp_targets.float(),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_disp_pos_avg

        loss = {
            "loss_dcr_disp": disp_loss,
        }

        return loss

    def dcr_losses(self, instances):

        num_classes = instances["cls"].pred_cls.size(1)
        assert num_classes == self.num_classes
        num_gpus = get_world_size()
        losses = {}

        losses.update(self.dcr_cls_losses(instances["cls"], num_gpus))
        losses.update(self.dcr_reg_losses(instances["reg"], num_gpus))
        #losses.update(self.dcr_disp_losses(instances["disp"], num_gpus))

        return instances, losses

    def predict_proposals(
            self, pred_result, locations, images, training_target=None
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
            "locations": locations,
            "strides": self.strides
        }
        bundle.update(pred_result)

        if training_target is not None:

            bundle["reg_target"] = training_target["reg"]["targets"]
            bundle["cls_target"] = training_target["cls"]["targets"]
            bundle["disp_target"] = training_target["disp"]["targets"]
            bundle["disp_pos_inds"] = training_target["disp"]["pos_inds"]
            bundle["reg_label"] = training_target["reg"]["labels"]
            bundle["cls_pos_inds"] = training_target["cls"]["pos_inds"]
            bundle["reg_pos_inds"] = training_target["reg"]["pos_inds"]
            bundle["reg_neg_inds"] = training_target["reg"]["neg_inds"]

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.

            boxes, analysis = self.forward_for_single_feature_map(
                per_bundle, images
            )
            if len(boxes):
                sampled_boxes.append(boxes)

                if "cls" in analysis:
                    cls_analysis.append(analysis["cls"])
                if "reg" in analysis:
                    reg_analysis.append(analysis["reg"])
                if "pt" in analysis:
                    pt_analysis.append(analysis["pt"])

                for per_im_sampled_boxes in sampled_boxes[-1]:
                    per_im_sampled_boxes.fpn_levels = per_bundle["locations"].new_ones(
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
            self, bundle, images
    ):

        results = []
        analysis = defaultdict(list)

        N, C, H, W = bundle["pred_cls"].shape

        # put in the same format as locations

        pred_cls = bundle["pred_cls"].view(N, C, H, W).permute(0, 2, 3, 1)
        pred_cls = pred_cls.reshape(N, -1, C).sigmoid()

        pred_reg = bundle["pred_reg"].view(N, 4, H, W).permute(0, 2, 3, 1) * bundle["strides"]
        pred_reg = pred_reg.reshape(N, -1, 4)

        pred_iou = bundle["pred_iou"].view(N, 1, H, W).permute(0, 2, 3, 1)
        pred_iou = pred_iou.reshape(N, -1).sigmoid()

        pred_disp = bundle["pred_disp"].view(N, 2, H, W)
        location = bundle["locations"]

        _, num_touch, pred_cent = self.iterate_disp([pred_disp])

        box_pred = cat([location - pred_reg[0,:,:2], location + pred_reg[0,:,2:]], dim=1)
        #num_target = bundle["reg_pos_inds"].unique().shape[0] - 1

        # calculate true positive, false positive, false negative of prediction
        if "cls_target" in bundle:
            cls_target = bundle["cls_target"]
            cls_pos_inds = bundle["cls_pos_inds"] != -1
            reg_target = bundle["reg_target"] * bundle["strides"]
            reg_pos_inds = bundle["reg_pos_inds"] != -1
            reg_neg_inds = bundle["reg_neg_inds"]
            disp_target = bundle["disp_target"]
            disp_pos_inds = bundle["disp_pos_inds"]

            disp_target = disp_target.reshape(N, H, W, 2).permute(0,3,1,2)
            disp_pos_inds = disp_pos_inds.reshape(N, H, W)
            self_dir = (disp_target ** 2).sum(dim=1) == 0
            
            true = torch.zeros_like(pred_cls).bool()
            true[:,cls_pos_inds,cls_target[cls_pos_inds]] = 1
            analysis["cls"].append(self.calc_stat(true, pred_cls > self.pre_nms_thresh))

            reg_trg = cat([location - reg_target[:,:2], location + reg_target[:,2:]],dim=1)
            reg_trg[reg_trg < 0] = 0
            box_pred[box_pred < 0] = 0
            iou_trg = matched_boxlist_iou(Boxes(reg_trg), Boxes(box_pred))
            analysis["pt"].append(self.calc_stat(reg_pos_inds, pred_iou > self.pre_nms_thresh))

            _, gt_touch, gt_cent = self.iterate_disp([disp_target],disp_pos_inds, self_dir, images[0])
            print(gt_cent)
            pred_cent = gt_cent
            pred_iou = iou_trg.unsqueeze(0)

        if "cls_target" in bundle:
            cls_id = bundle["cls_pos_inds"].unique()[1:]
            reg_id = bundle["reg_pos_inds"].unique()[1:]
            if (len(cls_id) != 0) and (len(cls_id) == len(reg_id)) and (cls_id == reg_id).all().item():
                cls_pos = (bundle["cls_pos_inds"].unsqueeze(1) == bundle["cls_pos_inds"].unique()[1:])
                reg_pos = (bundle["reg_pos_inds"].unsqueeze(1) == bundle["reg_pos_inds"].unique()[1:])
                box_cls = []
                box_class = []
                pred_box = []
                box_location = []
                for j in range(len(cls_id)):
                    per_box_cls, per_class = pred_cls[0,cls_pos[:,j],:].max(dim=1)
                    t = per_box_cls.argmax()
                    box_cls.append(per_box_cls[t])
                    box_class.append(per_class[t])
                    t = iou_trg[reg_pos[:,j]].argmax()
                    pred_box.append(box_pred[reg_pos[:,j],:][t].unsqueeze(0))
                    box_location.append(location[reg_pos[:,j],:][t].unsqueeze(0))

                box_cls = torch.stack(box_cls)
                box_class = torch.stack(box_class)
                pred_box = cat(pred_box, dim=0)
                box_location = cat(box_location, dim=0)

                boxlist = Instances(images.image_sizes[0])
                boxlist.pred_boxes = Boxes(pred_box)
                boxlist.scores = box_cls
                boxlist.pred_classes = box_class
                boxlist.locations = box_location
                results.append(boxlist)
                return results, analysis

            candidate_inds = (gt_touch[0] > 1).nonzero()
            #candidate_inds = box_logits > self.pre_nms_thresh
        else:
            candidate_inds = (num_touch[0] > 1).nonzero()

        if len(candidate_inds):
            dist = ((pred_cent[0].permute(0,2,3,1).reshape(-1,1,3) - candidate_inds[:,[0,2,1]]) ** 2)
            candidate_inds = (-dist.sum(dim=2)).topk(30,dim=0)[1]
            #iou_trg, reg_idx = iou_trg[candidate_inds].max(dim=0)
            _, reg_idx = pred_iou[0,candidate_inds].max(dim=0)
            box_pred = box_pred[candidate_inds][reg_idx,torch.arange(len(reg_idx)),:]
            per_box_cls, per_class = pred_cls[:,candidate_inds,:].max(dim=1)[0].max(dim=-1)
            per_locations = location[candidate_inds][reg_idx, torch.arange(len(reg_idx)), :]

            for i in range(N):
                boxlist = Instances(images.image_sizes[i])
                boxlist.pred_boxes = Boxes(box_pred)
                boxlist.scores = per_box_cls[i]
                boxlist.pred_classes = per_class[i]
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

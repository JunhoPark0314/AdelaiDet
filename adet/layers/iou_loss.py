import torch
from torch import nn


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()

class IOUExtendLoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOUExtendLoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_bottom = pred[:, 1]
        pred_right = pred[:, 2]
        pred_top = pred[:, 3]

        target_left = target[:, 0]
        target_bottom = target[:, 1]
        target_right = target[:, 2]
        target_top = target[:, 3]

        assert (target_left <= target_right).all().item() and (target_bottom <= target_top).all().item()
        assert (pred_left <= pred_right).all().item() and (pred_bottom <= pred_top).all().item()

        target_aera = (target_right - target_left) * \
                      (target_top - target_bottom)
        pred_aera = (pred_right - pred_left) * \
                      (pred_top - pred_bottom)

        w_intersect = torch.clamp(
                            torch.min(pred_right, target_right) - \
                            torch.max(pred_left, target_left) ,
                            min=0)
                      
        h_intersect = torch.clamp(
                            torch.min(pred_top, target_top) - \
                            torch.max(pred_bottom, target_bottom),
                            min=0)

        g_w_intersect = torch.max(pred_right, target_right) - \
                        torch.min(pred_left, target_left)

        g_h_intersect = torch.max(pred_top, target_top) - \
                        torch.min(pred_bottom, target_bottom)


        assert torch.logical_and((g_w_intersect >= 0).all(), (g_h_intersect >= 0).all()).item()
        assert torch.logical_and((w_intersect >= 0).all(), (h_intersect >= 0).all()).item()
        assert torch.logical_and((target_aera >= 0).all(), (pred_aera >= 0).all()).item()

        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        #assert (ac_uion >= area_union).all().item()

        ious = (area_intersect + 1e-6) / (area_union + 1e-6)
        gious = ious - (ac_uion - area_union) / ac_uion

        losses = 1 - gious

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()

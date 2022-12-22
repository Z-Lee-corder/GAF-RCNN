import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils


class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )
        self.iou_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.IOU_FC,
            input_channels=input_channels,
            output_channels=1
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_seg, tb_dict_1 = self.get_seg_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        point_loss_cls, tb_dict_3 = self.get_cls_layer_loss()
        #point_loss_iou, tb_dict_4 = self.get_iou_layer_loss()

        point_loss = point_loss_seg + point_loss_box + point_loss_cls #+ point_loss_iou
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_3)
        #tb_dict.update(tb_dict_4)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        #point_iou_preds = self.iou_layers(point_features)  # (total_points, 1)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            #ret_dict['point_iou_preds'] = point_iou_preds

        point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )

        batch_dict['batch_cls_preds'] = point_cls_preds
        batch_dict['batch_box_preds'] = point_box_preds
        batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
        batch_dict['cls_preds_normalized'] = False

        if self.training:
            point_cls_scores_loss, point_cls_lables_loss = torch.max(point_cls_preds, dim=1)
            point_cls_lables_loss = (point_cls_lables_loss + 1)
            point_cls_scores_loss = (torch.sigmoid(point_cls_scores_loss))
            point_box_preds_loss = point_box_preds
            gt_boxes = batch_dict['gt_boxes'].view(-1, 8)
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=point_box_preds_loss, roi_labels=point_cls_lables_loss,
                gt_boxes=gt_boxes[:, 0:7], gt_labels=gt_boxes[:, -1].long()
            )

            box_ious = max_overlaps
            gt_of_boxes = gt_boxes[gt_assignment]
            if self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'cls':
                box_cls_labels = (box_ious > self.model_cfg.TARGET_CONFIG.CLS_FG_THRESH).long()
                ignore_mask = (box_ious > self.model_cfg.TARGET_CONFIG.CLS_BG_THRESH) & \
                              (box_ious < self.model_cfg.TARGET_CONFIG.CLS_FG_THRESH)
                box_cls_labels[ignore_mask > 0] = -1
            elif self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'roi_iou':
                iou_bg_thresh = self.model_cfg.TARGET_CONFIG.CLS_BG_THRESH
                iou_fg_thresh = self.model_cfg.TARGET_CONFIG.CLS_FG_THRESH
                fg_mask = box_ious > iou_fg_thresh
                bg_mask = box_ious < iou_bg_thresh
                interval_mask = (fg_mask == 0) & (bg_mask == 0)

                box_cls_labels = (fg_mask > 0).float()
                box_cls_labels[interval_mask] = \
                    (box_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

            ret_dict['point_cls_scores_loss'] = point_cls_scores_loss
            ret_dict['box_cls_labels'] = box_cls_labels
            ret_dict['predict_boxes'] = point_box_preds_loss
            ret_dict['gt_of_boxes'] = gt_of_boxes[:, 0:7]
            ret_dict['box_ious'] = box_ious
            self.forward_ret_dict = ret_dict
        #else:
            #batch_dict['point_cls_scores'] = batch_dict['point_cls_scores'] * (point_iou_preds ** 4)

        return batch_dict

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment

from typing import Dict, List, Literal
import numpy as np
import torch
import torch.nn as nn
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.utils import class2angle
from utils import box_ops


class BBoxCoder:
    def decode(self,
               outputs: Dict[str, torch.Tensor],
               info: Dict[str, np.ndarray],
               calibs: List[Calibration],
               cls_mean_size: np.ndarray,
               threshold: float = 0.2,
               topk: int = 50) -> Dict[str, np.ndarray]:
        dets = self.extract_dets_from_outputs(outputs, topk)
        dets = dets.detach().cpu().numpy()
        results = self.decode_detections(dets, info, calibs, cls_mean_size, threshold)
        return results

    def get_heading_angle(self, heading):
        heading_bin, heading_res = heading[0:12], heading[12:24]
        heading_cls = heading_bin.argmax()
        res = heading_res[heading_cls]
        return class2angle(heading_cls, res, to_label_format=True)

    def decode_detections(self,
                          dets: np.ndarray,
                          info: Dict[str, np.ndarray],
                          calibs: List[Calibration],
                          cls_mean_size: np.ndarray,
                          threshold: float,
                          ) -> Dict[str, np.ndarray]:
        '''
        NOTE: THIS IS A NUMPY FUNCTION
        input: dets, numpy array, shape in [batch x max_dets x dim]
        input: img_info, dict, necessary information of input images
        input: calibs, corresponding calibs for the input batch
        output:
        '''
        results = {}
        for i in range(dets.shape[0]):  # batch
            preds = []
            for j in range(dets.shape[1]):  # max_dets
                cls_id = int(dets[i, j, 0])
                score = dets[i, j, 1]
                if score < threshold:
                    continue

                # 2d bboxs decoding
                x = dets[i, j, 2] * info['img_size'][i][0]
                y = dets[i, j, 3] * info['img_size'][i][1]
                w = dets[i, j, 4] * info['img_size'][i][0]
                h = dets[i, j, 5] * info['img_size'][i][1]
                bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                # 3d bboxs decoding
                # depth decoding
                depth = dets[i, j, 6]

                # dimensions decoding
                dimensions = dets[i, j, 31:34]
                dimensions += cls_mean_size[int(cls_id)]

                # positions decoding
                x3d = dets[i, j, 34] * info['img_size'][i][0]
                y3d = dets[i, j, 35] * info['img_size'][i][1]
                locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                # heading angle decoding
                alpha = self.get_heading_angle(dets[i, j, 7:31])
                ry = calibs[i].alpha2ry(alpha, x)

                score = score * dets[i, j, -1]
                preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
            results[info['img_id'][i]] = preds
        return results

    def extract_dets_from_outputs(self, outputs: Dict[str, torch.Tensor], topk: int = 50) -> torch.Tensor:
        # get src outputs

        # b, q, c
        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

        # final scores
        scores = topk_values
        # final indexes
        topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
        # final labels
        labels = topk_indexes % out_logits.shape[2]

        heading = outputs['pred_angle']
        size_3d = outputs['pred_3d_dim']
        depth = outputs['pred_depth'][:, :, 0: 1]
        log_sigma = outputs['pred_depth'][:, :, 1: 2]
        sigma_inverse = torch.exp(-log_sigma)

        # decode
        boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

        xs3d = boxes[:, :, 0: 1]
        ys3d = boxes[:, :, 1: 2]

        heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
        depth = torch.gather(depth, 1, topk_boxes)
        sigma_inverse = torch.gather(sigma_inverse, 1, topk_boxes)
        size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

        corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

        xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
        size_2d = xywh_2d[:, :, 2: 4]

        xs2d = xywh_2d[:, :, 0: 1]
        ys2d = xywh_2d[:, :, 1: 2]

        batch = out_logits.shape[0]
        labels = labels.view(batch, -1, 1)
        scores = scores.view(batch, -1, 1)
        xs2d = xs2d.view(batch, -1, 1)
        ys2d = ys2d.view(batch, -1, 1)
        xs3d = xs3d.view(batch, -1, 1)
        ys3d = ys3d.view(batch, -1, 1)

        # encoder              [1     + 1     + 1   + 1   + 2      + 1    + 24     + 3      + 1   + 1   + 1    ]
        detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma_inverse], dim=2)

        return detections


def decode_detections(detections: List[Dict[str, np.ndarray]],
                      info: Dict[str, np.ndarray],
                      calibs: List[Calibration],
                      cls_mean_size: np.ndarray,
                      threshold: float,
                      ) -> Dict[str, np.ndarray]:
    '''Decodes list of detection dicts into KITTI format.
    NOTE: THIS IS A NUMPY FUNCTION
    
    detections: List of array dicts with keys:
        * labels: [num_boxes, 1]
        * scores: [num_boxes, 1]
        * x_2d: [num_boxes, 1]
        * y_2d: [num_boxes, 1]
        * w: [num_boxes, 1]
        * h: [num_boxes, 1]
        * size_3d: [num_boxes, 3]
        * x_3d: [num_boxes, 1]
        * y_3d: [num_boxes, 1]
        * depth: [num_boxes, 2]
        * alpha_angle: [num_boxes, 1]
    info: img info with keys:
        * img_size
        * img_id
    calibs: corresponding calibs for the input batch
    cls_mean_size: ndarray with shape [C, 3], where C is the number of classes
    output:
    '''
    results = {}
    for detection, img_size, img_id, calib in zip(detections, info['img_size'], info['img_id'], calibs):
        mask = (detection['scores'] >= threshold).squeeze()
        if np.all(~mask):
            results[img_id] = []
            continue

        detection = {key: val[mask] for key, val in detection.items()}

        labels = detection['labels'].astype(np.int32)
        scores = detection['scores']

        x_2d = detection['x_2d'] * img_size[0]
        y_2d = detection['y_2d'] * img_size[1]
        w = detection['w'] * img_size[0]
        h = detection['h'] * img_size[1]
        bbox = np.concatenate([x_2d - w / 2, y_2d - h / 2, x_2d + w / 2, y_2d + h / 2], axis=-1)

        size_3d = detection['size_3d'] + cls_mean_size[labels.ravel()]

        x_3d = detection['x_3d'] * img_size[0]
        y_3d = detection['y_3d'] * img_size[1]
        depth = detection['depth']
        locations = calib.img_to_rect(x_3d, y_3d, depth)
        locations[:, 1] += size_3d[:, 0] / 2

        alpha_angle = detection['alpha_angle']
        ry = calib.alpha2ry(alpha_angle, x_2d)

        preds = np.concatenate([labels,
                                alpha_angle,
                                bbox,
                                size_3d,
                                locations,
                                ry,
                                scores], axis=-1)
        results[img_id] = preds.tolist()
    return results


class OQDDBBoxCoder:
    def decode(self,
               outputs: Dict[str, torch.Tensor],
               info: Dict[str, np.ndarray],
               calibs: List[Calibration],
               cls_mean_size: np.ndarray,
               threshold: float = 0.2,
               topk: int = 50) -> Dict[str, np.ndarray]:
        dets = self.extract_dets_from_outputs(outputs, topk)
        # dict of arrays to list of dicts
        dets = [dict(zip(dets.keys(), arrays)) for arrays in zip(*dets.values())]
        results = decode_detections(dets, info, calibs, cls_mean_size, threshold)
        return results

    def get_heading_angle(self, heading: torch.Tensor) -> torch.Tensor:
        """Gets heading angle from bins and residuals.

        Args:
            heading: A tensor with shape [*, 24].

        Returns:
            A tensor with shape [*, 1] of angles in [-pi, pi].
        """
        heading_bin, heading_res = heading[..., :12], heading[..., 12:]
        heading_class = heading_bin.argmax(-1, keepdim=True)
        heading_residual = heading_res.gather(dim=-1, index=heading_class)
        return class2angle(heading_class, heading_residual, to_label_format=True)

    def extract_dets_from_outputs(self, outputs: Dict[str, torch.Tensor], topk: int = 50) -> Dict[str, np.ndarray]:
        detections: Dict[str, torch.Tensor] = {}

        # b, q, c
        out_logits = outputs['pred_logits']
        prob = out_logits.sigmoid()
        # [batch, num_boxes * num_classes]
        topk_values, topk_indexes = torch.topk(prob.flatten(1), topk, dim=1)

        # final indexes [batch, topk, 1]
        topk_box_indexes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)

        # final labels [batch, topk, 1]
        labels = (topk_indexes % out_logits.shape[2]).unsqueeze(-1)
        detections['labels'] = labels

        # [batch, topk, 1]
        heading = self.get_heading_angle(outputs['pred_angle'])
        detections['alpha_angle'] = heading.gather(1, topk_box_indexes)
        # [batch, topk, 3]
        detections['size_3d'] = outputs['pred_3d_dim'].gather(1, topk_box_indexes.repeat(1, 1, 3))

        # final depth [batch, topk, 2]
        pred_depth = outputs['pred_depth'].gather(1, topk_box_indexes.repeat(1, 1, 2))
        detections['depth'] = pred_depth[:, :, 0:1]

        depth_score = pred_depth[:, :, 1:2]
        # final scores [batch, topk, 1]
        # scores = topk_values.unsqueeze(-1) * depth_score
        # detections['scores'] = scores
        detections['scores'] = topk_values.unsqueeze(-1)

        # [batch, topk, 6]
        boxes = torch.gather(outputs['pred_boxes'], 1, topk_box_indexes.repeat(1, 1, 6))

        detections['x_3d'] = boxes[:, :, 0:1]
        detections['y_3d'] = boxes[:, :, 1:2]

        corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

        xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
        detections['x_2d'], detections['y_2d'], detections['w'], detections['h'] = xywh_2d.split(1, dim=-1)

        detection_dict_numpy = {key: tensor.detach().cpu().numpy() for key, tensor in detections.items()}
        return detection_dict_numpy


_AVAILABLE_BBOX_CODERS = {
    'BBoxCoder': BBoxCoder,
    'OQDDBBoxCoder': OQDDBBoxCoder,
}


def build_bbox_coder(cfg):
    if 'bbox_coder' in cfg:
        bbox_coder_type: str = cfg['bbox_coder'].pop('type', 'BBoxCoder')
        assert bbox_coder_type in _AVAILABLE_BBOX_CODERS, (
            f'Invalid bbox_coder type {bbox_coder_type}. Supported bbox_coder types are {list(_AVAILABLE_BBOX_CODERS.keys())}.')
        return _AVAILABLE_BBOX_CODERS[bbox_coder_type](**cfg['bbox_coder'])
    return BBoxCoder()
############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim = feat.size(2)  # get channel dim
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat

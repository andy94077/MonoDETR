import os
import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append('/home/team/MonoDETR/')

import copy
import random
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision.ops import roi_align

import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import (
    get_distance_eval_result, get_official_eval_result)
from lib.datasets.kitti.kitti_utils import (Calibration, affine_transform,
                                            get_affine_transform,
                                            get_objects_from_label)
from lib.datasets.utils import (angle2class, draw_umich_gaussian,
                                gaussian_radius)
from lib.datasets.kitti.pd import PhotometricDistort
from utils import box_ops, depth_utils


ImageFile.LOAD_TRUNCATED_IMAGES = True


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg['root_dir']
        self.split = split
        self.num_classes = 3
        self.max_objs = cfg.get('max_objs', 50)
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)
        self.use_gt_depth_map = cfg.get('use_gt_depth_map', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.depth_map_dir = os.path.join(self.data_dir, 'depth_dense')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191462, 1.62856739989, 3.88311640418],
                                       [1.73698127, 0.59706367, 1.76282397]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # depth
        self.depth_min = cfg.get('depth_min', 1e-3)
        self.depth_max = cfg.get('depth_max', 60)
        self.depth_map_downsample = cfg.get('depth_map_downsample', 16)

        self.with_roi_depth = cfg.get('with_roi_depth', False)
        self.grid_H = cfg.get('grid_H', 5)
        self.grid_W = cfg.get('grid_W', 7)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_depth_map(self, idx, img_size):
        depth_map = cv2.imread(os.path.join(self.depth_map_dir, f'{idx:06d}.png'), cv2.IMREAD_GRAYSCALE)
        dst_W, dst_H = img_size
        pad_h, pad_w = dst_H - depth_map.shape[0], (dst_W - depth_map.shape[1]) // 2
        pad_wr = dst_W - pad_w - depth_map.shape[1]
        depth_map = np.pad(depth_map, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
        return Image.fromarray(depth_map)

    def build_sparse_depth_map(self, depth_map_resolution: Sequence[int], boxes2d: torch.Tensor, center_depth: torch.Tensor) -> torch.Tensor:
        """Builds a sparse depth map from some 2D bboxes and their center depth.

        Args:
            depth_map_resolution: The depth map size (W, H).
            boxes2d: A tensor of normalized 2D bboxes (cx, cy, w, h) with shape [num_boxes, 4]. Each element is in [0, 1].
            center_depth: A tensor representing the center depth in 3D space of each 2D bboxes with shape [num_boxes, 1].
        """
        W, H = depth_map_resolution
        boxes2d = boxes2d * boxes2d.new_tensor([W, H, W, H])
        # (x_min, y_min, x_max, y_max)
        boxes2d = box_ops.box_cxcywh_to_xyxy(boxes2d)
        # [num_boxes,]
        center_depth = center_depth.squeeze(dim=1)

        depth_map = boxes2d.new_full((H, W), self.depth_max)

        # Set box corners
        boxes2d[:, :2] = torch.floor(boxes2d[:, :2])
        boxes2d[:, 2:] = torch.ceil(boxes2d[:, 2:])
        boxes2d = boxes2d.long()

        # Set all values within each box to True
        center_depth, sorted_idx = torch.sort(center_depth, descending=True)
        boxes2d = boxes2d[sorted_idx]
        for bbox, depth in zip(boxes2d, center_depth):
            u1, v1, u2, v2 = bbox
            depth_map[v1:v2, u1:u2] = depth
        depth_map[(depth_map < self.depth_min) | (depth_map > self.depth_max)] = self.depth_max
        return depth_map

    def build_roi_depth(self, depth_map: torch.Tensor, boxes2d: torch.Tensor, center_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds roi aligned depth map from each 2D bboxes.

        Args:
            depth_map: A tensor of the depth map with shape [H, W].
            boxes2d: A tensor of normalized 2D bboxes (cx, cy, w, h) with shape [num_boxes, 4]. Each element is in [0, 1].
            center_depth: A tensor representing the center depth in 3D space of each 2D bboxes with shape [num_boxes, 1].
        """
        H, W = depth_map.shape
        boxes2d = boxes2d * boxes2d.new_tensor([W, H, W, H])
        # (x_min, y_min, x_max, y_max)
        boxes2d = box_ops.box_cxcywh_to_xyxy(boxes2d)
        # [num_boxes,]
        center_depth = center_depth.squeeze(dim=1)

        bbox_masks = (boxes2d[:, 0] < boxes2d[:, 2]) & (boxes2d[:, 1] < boxes2d[:, 3])

        roi_depths = torch.full((boxes2d.shape[0], self.grid_H, self.grid_W), self.depth_max, dtype=torch.float32)
        if bbox_masks.any():
            roi_aligned_out = roi_align(depth_map.unsqueeze(0).unsqueeze(0).type(torch.float32),
                                        [boxes2d[bbox_masks]], (self.grid_H, self.grid_W), aligned=True)
            roi_depths[bbox_masks] = roi_aligned_out[:, 0]

        # maintain interested points
        roi_depth_masks = torch.zeros((boxes2d.shape[0], self.grid_H, self.grid_W), dtype=bool)
        roi_depth_masks = (center_depth.view(-1, 1, 1) - 3 < roi_depths) & \
            (roi_depths < center_depth.view(-1, 1, 1) + 3) & \
            (roi_depths > 0)
        roi_depths[~roi_depth_masks] = self.depth_max

        return roi_depths, roi_depth_masks

    def eval(self, results_dir, logger) -> Tuple[Dict[str, float], float]:
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        class_to_eval = [test_id[category] for category in self.writelist]
        logger.info('==> Evaluating (official) ...')
        results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, class_to_eval)

        logger.info(results_str)
        # Returns the result dict and Car_3d_moderate
        return results_dict, mAP3d_R40

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False

        if self.data_augmentation:

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    w_shift = np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    h_shift = np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[0] += img_size[0] * w_shift
                    center[1] += img_size[1] * h_shift

        # add affine transformation for 2d images. Each matrix has shape [2, 3].
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        # # [3, 3]
        calib = self.get_calib(index)
        trans_for_calib = np.concatenate([trans, np.array([[0, 0, 1]])], axis=0, dtype=np.float32)
        calib_matrix = trans_for_calib @ calib.P2.copy()

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}

        if self.split == 'test':
            return img, calib_matrix, dict(), info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)

        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:
                    object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi:
                    object.alpha += 2 * np.pi
                if object.ry > np.pi:
                    object.ry -= 2 * np.pi
                if object.ry < -np.pi:
                    object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            # bbox_2d: (x_min, y_min, x_max, y_max)
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)
            # center_3d: [x, y, z] in camera coordinate -> [u, v, d]

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if not proj_inside_img:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            # Normalizes to almost [0, 1] (input sizes may vary)
            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = calib_matrix

        if self.use_gt_depth_map:
            depth_map = self.get_depth_map(index, img_size)
            if random_flip_flag:
                depth_map = depth_map.transpose(Image.FLIP_LEFT_RIGHT)

            # [depth_map_W, depth_map_H]
            final_depth_map_size = self.resolution // self.depth_map_downsample
            depth_map = depth_map.transform(tuple(self.resolution),
                                            method=Image.AFFINE,
                                            data=tuple(trans_inv.reshape(-1)),
                                            resample=Image.BILINEAR)

            depth_map_original = np.array(depth_map).astype(np.float32)
            depth_map = cv2.resize(depth_map_original, tuple(final_depth_map_size), interpolation=cv2.INTER_AREA)
            depth_map[(depth_map < self.depth_min) | (depth_map > self.depth_max)] = self.depth_max
            depth_map_original[(depth_map_original < self.depth_min) | (depth_map_original > self.depth_max)] = self.depth_max
            depth_map = torch.from_numpy(depth_map)
            depth_map_original = torch.from_numpy(depth_map_original)
        else:
            depth_map = self.build_sparse_depth_map(self.resolution // self.depth_map_downsample, torch.from_numpy(boxes), torch.from_numpy(depth))
            depth_map_original = self.build_sparse_depth_map(self.resolution, torch.from_numpy(boxes), torch.from_numpy(depth))

        if self.with_roi_depth:
            roi_depths, depth_masks = self.build_roi_depth(depth_map, torch.from_numpy(boxes), torch.from_numpy(depth))

        # collect return data
        inputs = img
        targets = {
            'calibs': calibs,
            'indices': indices,
            'img_size': img_size,
            'labels': labels,
            'boxes': boxes,  # normalized (cx, cy, w, h)
            'boxes_3d': boxes_3d,  # normalized (3d_cx, 3d_cy, l, r, t, b)
            'depth': depth,
            'size_2d': size_2d,  # (w, h)
            'size_3d': size_3d,  # real_size_3d - mean_size
            'src_size_3d': src_size_3d,  # real_size_3d
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'depth_map': depth_map,
            'depth_map_original': depth_map_original,  # for debuging only
            'mask_2d': mask_2d}
        if self.with_roi_depth:
            targets['roi_depth'] = roi_depths
            targets['depth_mask'] = depth_masks

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, calib_matrix, targets, info


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    cfg = {'root_dir': '/mnt/ssd2/KITTIDataset',
           'aug_crop': False, 'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.4, 'shift': 0.25, 'use_dontcare': False,
           'use_gt_depth_map': False, 'depth_min': 1e-3, 'depth_max': 60.0,
           'with_roi_depth': True,
           'class_merging': False, 'writelist': ['Car'], 'use_3d_center': True}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, calibs, targets, info) in enumerate(dataloader):
        if info['img_id'][0] != 10:
            continue
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray((img * dataset.std + dataset.mean) * 255, dtype=np.uint8)
        # print(targets['size_3d'][0][0])
        boxes = box_ops.box_cxcywh_to_xyxy(targets['boxes'][0])
        for x_min, y_min, x_max, y_max in boxes:
            x_min = x_min * 1280
            y_min = y_min * 384
            x_max = x_max * 1280
            y_max = y_max * 384
            img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
        img = Image.fromarray(img.astype(np.uint8))
        img.save('/home/team/MonoDETR/test.png')
        fig, ax = plt.subplots(figsize=(10, 3.2))
        depth_map = targets['depth_map'][0].numpy()
        ax.imshow(depth_map, vmin=cfg['depth_min'], vmax=cfg['depth_max'])
        # ax.imshow(targets['depth_map_original'][0].numpy(), vmin=cfg['depth_min'], vmax=cfg['depth_max'])
        # ax.set_xticks(np.arange(0, 1280, 100))
        fig.savefig('/home/team/MonoDETR/test_depth_map.png', bbox_inches='tight')
        num_roi_depth = 0
        for depth, roi_depth, box in zip(targets['depth'][0], targets['roi_depth'][0], boxes):
            if depth == 0:
                continue
            if num_roi_depth >= 10:
                break
            plt.figure(figsize=(4, 3))
            plt.imshow(roi_depth.numpy(), vmin=cfg['depth_min'], vmax=cfg['depth_max'])
            plt.title(f'{box.numpy() * np.array([depth_map.shape[1], depth_map.shape[0], depth_map.shape[1], depth_map.shape[0]])}')
            plt.savefig(f'/home/team/MonoDETR/test_roi_depth_{num_roi_depth}.png', bbox_inches='tight')
            num_roi_depth += 1

        # print ground truth fisrt
        objects = dataset.get_label(info['img_id'][0])
        for object in objects:
            print(object.to_kitti_format())
        break


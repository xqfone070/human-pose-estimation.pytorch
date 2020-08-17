# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import numpy as np
import json

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms


logger = logging.getLogger(__name__)


# data_root
#   |--train
#       |--images
#       |--annotations
#   |--val
#       |--images
#       |--annotations

class LabelmeDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.num_joints = 10
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.parent_ids = None

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        image_dir = os.path.join(self.root, self.image_set, 'images')
        annotation_dir = os.path.join(self.root, self.image_set, 'annotations')
        files = os.listdir(annotation_dir)
        recs = []
        for file in files:
            annotation_file = os.path.join(annotation_dir, file)
            filebase = os.path.splitext(file)[0]
            image_file = os.path.join(image_dir, filebase + '.jpg')
            rec = self._load_annotation(image_file, annotation_file)
            recs.append(rec)
        return recs

    # need double check this API and classes field
    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
            return {'Null': 0}, 0

    def _load_annotation(self, img_file, annotation_file):
        with open(annotation_file, encoding='utf-8') as f:
            data = json.load(f)

        joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)

        shape_data = data['shapes']
        landmark_label_prefix = 'landmark_'
        for shape in shape_data:
            label = shape['label']
            if label.startswith(landmark_label_prefix):
                landmark_index = int(label[len(landmark_label_prefix):])
                joints_3d[landmark_index, 0] = shape['points'][0][0]
                joints_3d[landmark_index, 1] = shape['points'][0][1]
                joints_3d[landmark_index, 2] = 0
                joints_3d_vis[landmark_index, 0] = 1
                joints_3d_vis[landmark_index, 1] = 1
                joints_3d_vis[landmark_index, 2] = 0
            elif label == 'pantograph':
                points = shape['points']
                xmin = points[0][0]
                ymin = points[0][1]
                xmax = points[1][0]
                ymax = points[1][1]

                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

        center, scale = self._box2cs(bbox)

        rec = {
                'image': img_file,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            }

        return rec


    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

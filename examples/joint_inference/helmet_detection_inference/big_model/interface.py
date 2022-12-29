# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import sys
import copy

import cv2
import numpy as np
import tensorflow as tf

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']


def draw_boxes(img, bboxes, colors, text_thickness, box_thickness):
    img_copy = copy.deepcopy(img)

    line_type = 2
    #  get color code
    colors = colors.split(",")
    colors_code = []
    for color in colors:
        if color == 'green':
            colors_code.append((0, 255, 0))
        elif color == 'blue':
            colors_code.append((255, 0, 0))
        elif color == 'yellow':
            colors_code.append((0, 255, 255))
        else:
            colors_code.append((0, 0, 255))

    label_dict = {i: label for i, label in enumerate(class_names)}

    for bbox in bboxes:
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(bbox[5])
        score = "%.2f" % round(bbox[4], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        try:
            cv2.rectangle(img_copy, p1[::-1], p2[::-1], colors_code[label],
                          box_thickness)
            cv2.putText(img_copy, text, (p1[1], p1[0] + 20 * (label + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        text_thickness, line_type)
        except TypeError as err:
            # error message from pyopencv,  cv2.circle only can accept centre
            # coordinates precision up to float32. If the coordinates are in
            # float64, it will throw this error.
            LOG.warning(f"Draw box fail: {err}")
    return img_copy


def output_deal(
        final_result,
        nframe,
        img_rgb
):
    # save and show image
    img_rgb = np.array(img_rgb).astype(np.float32)
    # cv2.imwrite(f"/data/tmp_222-{nframe}.jpeg", img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    collaboration_frame = draw_boxes(img_rgb, final_result,
                                     colors="green,blue,yellow,red",
                                     text_thickness=None,
                                     box_thickness=None)

    cv2.imwrite(f"/data/tmp_{nframe}.jpeg", collaboration_frame)


def preprocess(image, input_shape):
    ih, iw = input_shape
    h, w, _ = image.shape
    org_img_shape = (w, h)

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image.astype(np.float32), (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    preprocessed_data = image_paded.astype(np.float32)[np.newaxis, :]
    return preprocessed_data, org_img_shape


def postprocess(data, org_img_shape):
    pred_sbbox, pred_mbbox, pred_lbbox = data[1], data[2], data[0]
    num_classes = 4
    score_threshold = 0.3
    input_size = 544
    pred_bbox = np.concatenate(
        [np.reshape(pred_sbbox, (-1, 5 + num_classes)),
         np.reshape(pred_mbbox, (-1, 5 + num_classes)),
         np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate(
        [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
         pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    org_w, org_h = org_img_shape
    resize_ratio = min(1.0 * input_size / org_w, 1.0 * input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2.
    dh = (input_size - resize_ratio * org_h) / 2.

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # clip some boxes those are out of range
    pred_coor = np.concatenate(
        [np.maximum(pred_coor[:, :2], [0, 0]),
         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # discard some invalidboxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))

    # discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    
    mask = score_mask
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    bboxes = np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    bboxes = nms(bboxes, 0.4)
    return bboxes


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
            boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bbox_ = best_bbox.tolist()

            # cast into int for cls
            best_bbox_[5] = int(best_bbox[5])

            best_bboxes.append(best_bbox_)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def create_input_feed(sess, img_data, new_image=None):
    input_feed = {}

    input_img_data = sess.graph.get_tensor_by_name('input/input_data:0')
    input_feed[input_img_data] = img_data

    return input_feed


def create_output_fetch(sess):
    """Create output fetch for edge model inference"""
    pred_sbbox = sess.graph.get_tensor_by_name('pred_sbbox/concat_2:0')
    pred_mbbox = sess.graph.get_tensor_by_name('pred_mbbox/concat_2:0')
    pred_lbbox = sess.graph.get_tensor_by_name('pred_lbbox/concat_2:0')

    output_fetch = [pred_sbbox, pred_mbbox, pred_lbbox]
    return output_fetch


class Estimator:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.session = tf.Session(graph=graph, config=config)
        self.input_shape = [544, 544]
        self.create_input_feed = create_input_feed
        self.create_output_fetch = create_output_fetch
        self.v_num = 0

    def load(self, model_url=""):
        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.gfile.FastGFile(model_url, 'rb') as handle:
                    LOG.info(f"Load model {model_url}, "
                             f"ParseFromString start .......")
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(handle.read())
                    LOG.info("ParseFromString end .......")

                    tf.import_graph_def(graph_def, name='')
                    LOG.info("Import_graph_def end .......")

        LOG.info("Import model from pb end .......")

    def predict(self, data, **kwargs):
        print("data size:", sys.getsizeof(data))
        img_data_np = np.array(data)
        print("img_data_np shape:", img_data_np.shape)
        new_image, shapes = preprocess(img_data_np, self.input_shape)
        print("new_image size:", sys.getsizeof(new_image))

        self.v_num += 1
        with self.session.as_default():
            input_feed = self.create_input_feed(
                self.session, new_image, img_data_np)
            output_fetch = self.create_output_fetch(self.session)
            output = self.session.run(output_fetch, input_feed)
            bbox = postprocess(output, shapes)
            output_deal(bbox, self.v_num, data)
            return bbox

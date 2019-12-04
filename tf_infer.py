#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : tf_infer.py
#   Author      : YunYang1994
#   Created date: 2019-12-04 15:56:52
#   Description :
#
#================================================================

import cv2
import utils
import numpy as np
from utils import anchors, classes
import tensorflow as tf

return_elements  = ["input/input_data:0", "conv_sbbox/BiasAdd:0", "conv_mbbox/BiasAdd:0", "conv_lbbox/BiasAdd:0"]
pb_file          = "./yolov3_voc.pb"
input_image_path = "./road.jpeg"
num_class        = len(classes)
strides          = [8, 16, 32]
input_size       = (608, 608)
graph            = tf.Graph()

# Output shapes expected by the post-processor
output_shapes = [(1, 19, 19, (num_class+5)*3), (1, 38, 38, (num_class+5)*3), (1, 76, 76, (num_class+5)*3)]

image = cv2.imread(input_image_path)
image_size = image.shape[:2]
image_data = np.expand_dims(utils.image_preporcess(image, input_size), 0)
image_data = np.array(image_data, dtype=np.float32, order='C')

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    tf_outputs = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

# tf_outputs = [output.reshape(shape) for output, shape in zip(tf_outputs, output_shapes)]
pred_bboxes = [utils.decode(tf_outputs[i], anchors[i], strides[i]) for i in range(3)]
pred_bboxes = np.concatenate([np.reshape(pred_bbox, (-1, 5+num_class)) for pred_bbox in pred_bboxes], axis=0)

bboxes = utils.postprocess_boxes(pred_bboxes, image_size, input_size[0], 0.3)
bboxes = utils.nms(bboxes, 0.5, method='nms')
image = utils.draw_bbox(image, bboxes)
cv2.imwrite("result.jpg", image)






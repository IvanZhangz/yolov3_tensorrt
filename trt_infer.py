#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : trt_infer.py
#   Author      : YunYang1994
#   Created date: 2019-12-05 13:22:49
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import utils
from utils import anchors, classes
import common
TRT_LOGGER = trt.Logger()


engine_file_path = "yolov3.engine"
uff_file_path = "yolov3_voc.uff"
input_image_path = "./road.jpg"
input_size = (608, 608)
num_class = len(classes)
strides = [8, 16, 32]

if not os.path.exists(uff_file_path): raise ValueError("%s does not exists!" %uff_file_path)

image = cv2.imread(input_image_path)
image_size = image.shape[:2]
image_data = np.expand_dims(utils.image_preporcess(image, input_size), 0)
image_data = np.array(image_data, dtype=np.float32, order='C')

# Output shapes expected by the post-processor
output_shapes = [(1, 19, 19, (num_class+5)*3), (1, 38, 38, (num_class+5)*3), (1, 76, 76, (num_class+5)*3)]

if not os.path.exists(engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        # parser.register_input("input/input_data", (3, 608, 608))
        parser.register_input("input/input_data", (608, 608, 3), trt.UffInputOrder.NHWC)
        parser.register_output("conv_sbbox/BiasAdd")
        parser.register_output("conv_mbbox/BiasAdd")
        parser.register_output("conv_lbbox/BiasAdd")
        parser.parse(uff_file_path, network)
        # Build and return an engine.
        print('Building an engine from file {}; this may take a while...'.format(uff_file_path))
        engine = builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

print("=> Reading engine from file {}".format(engine_file_path))
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Do inference
    print('Running inference on image {}...'.format(input_image_path))
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = image_data
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
pred_bboxes = [utils.decode(trt_outputs[i], anchors[i], strides[i]) for i in range(3)]
pred_bboxes = np.concatenate([np.reshape(pred_bbox, (-1, 5+num_class)) for pred_bbox in pred_bboxes], axis=0)

bboxes = utils.postprocess_boxes(pred_bboxes, image_size, input_size[0], 0.4)
bboxes = utils.nms(bboxes, 0.5, method='nms')
image = utils.draw_bbox(image, bboxes)
cv2.imwrite("trt_result.jpg", image)



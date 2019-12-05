This toy project leads you to explore how to export YOLO model and deploy them on the TensorRT plattform. 

## requirements

- tensorrt==5.1.5.0
- tensorflow==1.10
- pycuda==2019.1.2

## useage
Download the yolov3_voc.pb from https://github.com/dl-playground/yolov3_tensorrt/releases/download/0.1/yolov3_voc.pb and put it in the current dir, then

```bashrc
$ convert-to-uff yolov3_voc.pb
$ python trt_infer.py
$ python tf_infer.py
```

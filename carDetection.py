import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.models import Model
import struct
import cv2
import tensorflow as tf

from car_detection.KerasYolo3.yolo3 import *

if __name__ == "__main__":
    video = cv2.VideoCapture('./lalala1.mp4')
    success, frame = video.read()
    if success:
        line_image = np.copy(frame) * 0

    writer = cv2.VideoWriter('./lalala2.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (1920, 1080))

    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader("./car_detection/KerasYolo3/yolov3.weights")
    weight_reader.load_weights(yolov3)
    px = None
    py = None

    errorness1 = 105
    errorness2 = 5
    while success:

        # frame = process_image(frame, line_image)
        writer.write(frame)

        success, frame = video.read()

        image_h, image_w, _ = frame.shape
        new_image = preprocess_input(frame, net_h, net_w)
        yolos = yolov3.predict(new_image)
        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        # draw bounding boxes on the image using labels
        _, px, py = draw_boxes(frame, boxes, labels, obj_thresh)

        frame = (frame).astype('uint8')

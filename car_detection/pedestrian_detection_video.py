import numpy as np
import cv2
import imageio

imageio.plugins.ffmpeg.download()
import imutils
import matplotlib.image as mpimg
from imutils.object_detection import non_max_suppression
from moviepy.editor import *


def pedestrian_detection_image(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    imshape = image.shape

    # image = mpimg.imread("xuezhang.jpeg")
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, hitThreshold=0.01, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    '''
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
    print("time:", end - start)
    '''

    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    # print ("finished")
    # cv2.waitKey(0)
    image = imutils.resize(image, width=imshape[1], height=imshape[0])

    return image


white_output = 'result4pedes.mp4'
clip1 = VideoFileClip("test5.mp4")
white_clip = clip1.fl_image(pedestrian_detection_image)
final_clip = clips_array([[clip1, white_clip]])
final_clip.write_videofile(white_output, audio=False)

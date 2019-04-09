import cv2
import numpy as np
import glob

if __name__=="__main__":
    video = cv2.VideoCapture("./data.mp4")
    success, frame = video.read()
    writer = cv2.VideoWriter("./lalala.avi", cv2.VideoWriter_fourcc('I', "4", '2', '0'), 20.0, (1920, 1080))
    cnt = 0

    while success:
        success, frame2 = video.read()
        tmp = frame2.copy()
        # writer.write(frame2)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary", binary)
        # cv2.waitKey(0)

        frame2[binary==0] = [0, 0, 0]
        # writer.write(frame2)
        for i in range(frame2.shape[0]):
            for j in range(frame2.shape[1]):
                isFlag = True
                for k in range(3):
                    if frame2[i, j, k] != 0:
                        isFlag = False
                        break
                    else:
                        continue

                if isFlag:
                    frame2[i, j] = frame[i, j]

        # cv2.imshow("frame", frame2)
        # cv2.waitKey(0)
        writer.write(frame2)
        frame = tmp
        print(cnt)
        cnt+=1

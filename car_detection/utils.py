import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# Before everything, make a difference
def preprocess(image):
    # t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lightness = np.mean(hsv[:, :, 2])
    var = np.std(hsv[:, :, 2])

    hue = np.mean(hsv[:, :, 0])

    if lightness < 95:
        gamma = 1 / 2.2
        little = image.astype(np.float32) / 255
        corrected1 = np.power(little, gamma)
        corrected1 = (corrected1 * 255).astype(np.uint8)
        print ("...Correcting Lightness...")
    else:
        corrected1 = image

    if var > 150:
        hsv2 = cv2.cvtColor(corrected1, cv2.COLOR_RGB2HSV)
        corrected2 = cv2.equalizeHist(hsv2[:, :, 2])
        corrected2 = cv2.cvtColor(corrected2, cv2.COLOR_HSV2RGB)
        print ("...Correcting Contrast...")
    else:
        corrected2 = corrected1

    if hue > 80:
        enhance_white = white_enhance(corrected2)
        enhance_yellow = yellow_enhance(corrected2)
        corrected3 = cv2.addWeighted(enhance_white, 0.8, enhance_yellow, 1, 0)
        print ("...Correcting Hue...")
    else:
        corrected3 = corrected2

    return corrected3


def detect_least(lines1):
    t1 = [0, 0]
    left = right = False
    norms = hough_filter(lines1)
    total = {}
    flag = False
    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                if total.keys() is None:
                    if m > 0:
                        t1[0] += 1
                    else:
                        t1[1] += 1
                    total[m] = [m * n, n, b * n, 1]
                else:
                    for c in total.keys():
                        if abs(c - m) <= 0.2:
                            total[c][0] += m * n
                            total[c][1] += n
                            total[c][2] += b * n
                            total[c][3] += 1
                            flag = True
                            break
                    if not flag:
                        if m > 0:
                            t1[0] += 1
                        else:
                            t1[1] += 1
                        total[m] = [m * n, n, b * n, 1]
                    else:
                        flag = False
    return t1[0] == 1, t1[1] == 1


# Detect lines in the image whether it's ok
def detect_lines(lines1, imshape):
    left = right = False
    norms = hough_filter(lines1)

    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                # y = mx + b
                if m > 0:
                    right = True
                if m < 0:
                    left = True
    return left, right


# Draw a average line on the board
def draw_lines(lines1, line_image, imshape):
    if lines1 is None:
        return None
    norms = hough_filter(lines1)

    total = {}
    m_total_right = 0
    n_total_right = 0
    m_total_left = 0
    n_total_left = 0
    b_total_right = 0
    b_total_left = 0

    flag = False
    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                # print m
                if total.keys() is None:
                    total[m] = [m * n, n, b * n, 1]
                else:
                    for c in total.keys():
                        if abs(c - m) <= 0.1:
                            total[c][0] += m * n
                            total[c][1] += n
                            total[c][2] += b * n
                            total[c][3] += 1
                            flag = True
                            break
                    if not flag:
                        total[m] = [m * n, n, b * n, 1]
                    else:
                        flag = False
                # print m
    # print total
    maxR = -1
    maxL = -1
    for i in total.keys():
        # print i
        if i > 0 and total[i][1] * total[i][3] > maxR:
            maxR = total[i][1] * total[i][3]
            m_total_right = total[i][0]
            n_total_right = total[i][1]
            b_total_right = total[i][2]
        elif i < 0 and total[i][1] * total[i][3] > maxL:
            maxL = total[i][1] * total[i][3]
            m_total_left = total[i][0]
            n_total_left = total[i][1]
            b_total_left = total[i][2]
    # print m_total_right, b_total_right, n_total_right

    if m_total_left != 0 or b_total_left != 0:
        b_left = b_total_left / n_total_left
        m_left = m_total_left / n_total_left
        # print b_avg_left, m_avg_left
        '''y = mx + b'''
        if b_left < imshape[0]:
            xa = 0
            ya = b_left
        else:
            ya = imshape[0]
            xa = (ya - b_left) / m_left
        ya2 = imshape[0] * 1.8 / 3
        xa2 = (ya2 - b_left) / m_left
        cv2.line(line_image, (int(xa), int(ya)), (int(xa2), int(ya2)), (255, 0, 0), 5)

    if m_total_right != 0 or b_total_right != 0:
        b_right = b_total_right / n_total_right
        m_right = m_total_right / n_total_right
        '''y = mx + b'''
        x_try = imshape[1]
        y_try = imshape[1] * m_right + b_right
        if y_try < imshape[0]:
            xb = x_try
            yb = y_try
        else:
            yb = imshape[0]
            xb = (yb - b_right) / m_right
        yb1 = imshape[0] * 1.8 / 3
        xb1 = (yb1 - b_right) / m_right
        cv2.line(line_image, (int(xb), int(yb)), (int(xb1), int(yb1)), (255, 0, 0), 5)

    return line_image


# Draw a average line on the board
def draw_lines_continuous(lines1, line_image, imshape):
    if lines1 is None:
        return None, 0, 0, 0, 0
    norms = hough_filter(lines1)

    total = {}
    m_total_right = 0
    n_total_right = 0
    m_total_left = 0
    n_total_left = 0
    b_total_right = 0
    b_total_left = 0
    m_left = 0
    b_left = 0
    m_right = 0
    b_right = 0
    flag = False
    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                if total.keys() is None:
                    total[m] = [m * n, n, b * n, 1]
                else:
                    for c in total.keys():
                        if abs(c - m) <= 0.2:
                            total[c][0] += m * n
                            total[c][1] += n
                            total[c][2] += b * n
                            total[c][3] += 1
                            flag = True
                            break
                    if not flag:
                        total[m] = [m * n, n, b * n, 1]
                    else:
                        flag = False
                # print m
    # print total
    maxR = -1
    maxL = -1
    for i in total.keys():
        if i > 0 and total[i][1] * total[i][3] > maxR:
            maxR = total[i][1] * total[i][3]
            m_total_right = total[i][0]
            n_total_right = total[i][1]
            b_total_right = total[i][2]
        elif i < 0 and total[i][1] * total[i][3] > maxL:
            maxL = total[i][1] * total[i][3]
            m_total_left = total[i][0]
            n_total_left = total[i][1]
            b_total_left = total[i][2]
    # print m_total_right, b_total_right, n_total_right

    if m_total_left != 0 or b_total_left != 0:
        b_left = b_total_left / n_total_left
        m_left = m_total_left / n_total_left
        # print b_avg_left, m_avg_left
        '''y = mx + b'''
        if b_left < imshape[0]:
            xa = 0
            ya = b_left
        else:
            ya = imshape[0]
            xa = (ya - b_left) / m_left
        ya2 = imshape[0] * 1.8 / 3
        xa2 = (ya2 - b_left) / m_left
        cv2.line(line_image, (int(xa), int(ya)), (int(xa2), int(ya2)), (255, 0, 0), 5)

    if m_total_right != 0 or b_total_right != 0:
        b_right = b_total_right / n_total_right
        m_right = m_total_right / n_total_right
        '''y = mx + b'''
        x_try = imshape[1]
        y_try = imshape[1] * m_right + b_right
        if y_try < imshape[0]:
            xb = x_try
            yb = y_try
        else:
            yb = imshape[0]
            xb = (yb - b_right) / m_right
        yb1 = imshape[0] * 1.8 / 3
        xb1 = (yb1 - b_right) / m_right
        cv2.line(line_image, (int(xb), int(yb)), (int(xb1), int(yb1)), (255, 0, 0), 5)

    return line_image, m_left, b_left, m_right, b_right


# draw line image based on m and b
def draw_lines_mb(ml, bl, mr, br, line_image, imshape):
    if bl < imshape[0]:
        xa = 0
        ya = bl
    else:
        ya = imshape[0]
        xa = (ya - bl) / (ml+0.0001)
    ya2 = imshape[0] * 1.8 / 3
    xa2 = (ya2 - bl) / (ml+0.0001)

    xt = imshape[1]
    yt = imshape[1] * mr + br
    if yt < imshape[0]:
        xb = xt
        yb = yt
    else:
        yb = imshape[0]
        xb = (yb - br) / (mr+0.0001)
    yb1 = imshape[0] * 1.8 / 3
    xb1 = (yb1 - br) / (mr+0.0001)

    cv2.line(line_image, (int(xb), int(yb)), (int(xb1), int(yb1)), (255, 0, 0), 5)
    cv2.line(line_image, (int(xa), int(ya)), (int(xa2), int(ya2)), (255, 0, 0), 5)

    return line_image


# a Hough Filter based on theta and similarity of the line
def hough_filter(lines):
    norms = {}
    flag = False
    if lines is None:
        return norms
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
            norm = [(math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2), fit[0], fit[1]), (x1, x2, y1, y2)]
            if norms.keys() is None:
                if (75 > theta) and (theta > 15):
                    norms[theta] = [norm]
            else:
                for t in norms.keys():
                    if (abs(t - theta) < 5) and (((float(norms[t][0][0][1]) - fit[0]) ** 2 + (
                            float(norms[t][0][0][2]) - fit[1]) ** 2) ** 0.5 < 20):
                        norms[t].append(norm)
                        flag = True
                        break
                if flag:
                    flag = False
                    continue
                else:
                    if (75 > theta) and (theta > 15):
                        norms[theta] = [norm]

    return norms


# Get the line's position to determine whether to wait or not
def get_lines(lines1, imshape):
    if lines1 is None:
        return None
    norms = hough_filter(lines1)

    m_total_right = 0
    n_total_right = 0
    m_total_left = 0
    n_total_left = 0
    b_total_right = 0
    b_total_left = 0
    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                if m > 0:
                    m_total_right += m * n
                    n_total_right += n
                    b_total_right += b * n
                else:
                    m_total_left += m * n
                    n_total_left += n
                    b_total_left += b * n

    if m_total_left != 0 or b_total_left != 0:
        left = True
        b_left = b_total_left / n_total_left
        m_left = m_total_left / n_total_left
        '''y = mx + b'''
        if b_left < imshape[0]:
            xa = 0
            ya = b_left
        else:
            ya = imshape[0]
            xa = (ya - b_left) / m_left
        ya2 = imshape[0] * 1.8 / 3
        xa2 = (ya2 - b_left) / m_left
    else:
        left = False

    if m_total_right != 0 or b_total_right != 0:
        right = True
        b_right = b_total_right / n_total_right
        m_right = m_total_right / n_total_right
        '''y = mx + b'''
        x_try = imshape[1]
        y_try = imshape[1] * m_right + b_right
        if y_try < imshape[0]:
            xb = x_try
            yb = y_try
        else:
            yb = imshape[0]
            xb = (yb - b_right) / m_right
        yb1 = imshape[0] * 1.8 / 3
        xb1 = (yb1 - b_right) / m_right
    else:
        right = False

    if left and right:
        return m_left, b_left, m_right, b_right
    elif left and not right:
        return m_left, b_left, 0, 0
    elif right and not left:
        return 0, 0, m_right, b_right
    else:
        return 0, 0, 0, 0


# finding the average line
def average_lines(lines, imshape):
    hough_pts = {'m_left': [], 'b_left': [], 'norm_left': [], 'm_right': [], 'b_right': [], 'norm_right': []}
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                m = fit[0]
                b = fit[1]
                # print m, b
                norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if m > 0:
                    hough_pts['m_right'].append(m)
                    hough_pts['b_right'].append(b)
                    hough_pts['norm_right'].append(norm)
                if m < 0:
                    hough_pts['m_left'].append(m)
                    hough_pts['b_left'].append(b)
                    hough_pts['norm_left'].append(norm)

    if len(hough_pts['b_left']) != 0 or len(hough_pts['m_left']) != 0 or len(hough_pts['norm_left']) != 0:
        b_avg_left = np.mean(np.array(hough_pts['b_left']))
        m_avg_left = np.mean(np.array(hough_pts['m_left']))
        # print b_avg_left, m_avg_left
        '''y = mx + b'''
        if b_avg_left < imshape[0]:
            xa = 0
            ya = b_avg_left
        else:
            ya = imshape[0]
            xa = (ya - b_avg_left) / m_avg_left
        ya2 = imshape[0] / 3
        xa2 = (ya2 - b_avg_left) / m_avg_left
        left_lane = [int(xa), int(ya), int(xa2), int(ya2)]
    else:
        left_lane = [0, 0, 0, 0]
    if len(hough_pts['b_right']) != 0 or len(hough_pts['m_right']) != 0 or len(hough_pts['norm_right']) != 0:
        b_avg_right = np.mean(np.array(hough_pts['b_right']))
        m_avg_right = np.mean(np.array(hough_pts['m_right']))
        '''y = mx + b'''
        if b_avg_right < imshape[0]:
            xb = 0
            yb = b_avg_right
        else:
            yb = imshape[0]
            xb = (yb - b_avg_right) / m_avg_right
        yb1 = imshape[0] / 3
        xb1 = (yb - b_avg_right) / m_avg_right
        right_lane = [int(xb1), int(yb1), int(xb), int(yb)]
    else:
        right_lane = [0, 0, 0, 0]

    return left_lane, right_lane


def abs_sobel_thresh(img, orient='x', thresh_min=90, thresh_max=255):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    binary_output = np.zeros_like(abs_sobel)
    binary_output[(abs_sobel >= thresh_min) & (abs_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


# Canny filter
def canny_thresh(img, low=20, high=90):
    canny = cv2.Canny(img, low, high)
    binary_out = np.zeros_like(canny)
    binary_out[canny == 255] = 1
    return binary_out


# HLS space filter
def hls_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, 2]
    binary_out = np.zeros_like(channel)
    binary_out[(channel > 150) & (channel <= 255)] = 1

    return binary_out


# LUV space filter
def luv_thresh(img, thresh=(160, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1

    return binary_output


# Yellow color filter
def thresh_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([18, 55, 46])
    upper = np.array([34, 180, 250])
    mask = cv2.inRange(hsv, lower, upper)

    return mask


# Enhance the yellow part of the image
def yellow_enhance(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([18, 80, 46])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return cv2.addWeighted(gray, 0.8, mask, 1, 0)


# Enhance the white part of the image
def white_enhance(img_rgb):
    lower_white = np.array([140, 140, 140])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img_rgb, lower_white, upper_white)

    return mask


# White color filter
def thresh_white(image):
    lower = np.array([140, 140, 140])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    return mask


# Judging whether lines are same in continuous algorithm
def isSame(ml, bl, mr, br):
    if abs(ml - mr) < 0.4:
        pass
    else:
        return False
    if abs(bl - br) < 50:
        return True
    else:
        return False

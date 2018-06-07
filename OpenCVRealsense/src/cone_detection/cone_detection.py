import cv2
import matplotlib.pyplot as plt
import numpy as np

from cone_detection.utilities import *


def copy_blank(img):
    height, width, channels = img.shape
    return np.zeros((height, width, channels), np.uint8)


# Each contour is an array of boundary points, each point enclosed in a redundant 1-size array.
# contour[0][0] is the first bounary point.
# contour[1][0] is the second bounary point.
# contour[1][0][1] is the 'y' coordinate of the second bounary point.
# Convex Hull passed in is a contour.
def hull_pointing_up(hull):
    x, y, w, h = cv2.boundingRect(hull)  # (x,y): top-left coordinate
    aspect_ratio = float(w) / h
    if aspect_ratio > 0.8:
        return False, [], [];

    yCenter = (y + h) / 2.0
    pointsAboveCenter = []
    pointsBelowCenter = []
    for point in hull:
        y = point[0][1]
        if y < yCenter:
            pointsAboveCenter.append(point)
        else:
            pointsBelowCenter.append(point)
    leftMostXBelowCenter = hull[0][0][0];
    rightMostXBelowCenter = hull[0][0][0];
    for point in pointsBelowCenter:
        x = point[0][0]
        if (x < leftMostXBelowCenter):
            leftMostXBelowCenter = x
            continue
        if (x > rightMostXBelowCenter):
            rightMostXBelowCenter = x
            continue

    for point in pointsAboveCenter:
        x = point[0][0]
        if (x < leftMostXBelowCenter or x > rightMostXBelowCenter):
            return False, pointsAboveCenter, pointsBelowCenter;

    leftMostXAboveCenter = hull[0][0][0];
    rightMostXAboveCenter = hull[0][0][0];
    for point in pointsAboveCenter:
        x = point[0][0]
        if (x < leftMostXAboveCenter):
            leftMostXAboveCenter = x
            continue
        if (x > rightMostXAboveCenter):
            rightMostXAboveCenter = x
            continue

    aboveWidth = rightMostXAboveCenter - leftMostXAboveCenter
    belowWidth = rightMostXBelowCenter - leftMostXBelowCenter
    shape_ratio = aboveWidth / belowWidth;
    if shape_ratio > 0.1:
        return False, pointsAboveCenter, pointsBelowCenter;

    return True, pointsAboveCenter, pointsBelowCenter;


def imshow_next(img, titleText):
    if imshow_next.show is False:
        return
    if imshow_next.ax_show is True:
        ax = imshow_next.ax;
        imshow_next.imgIdx = imshow_next.imgIdx + 1
        idx = imshow_next.imgIdx

        ax[idx].imshow(img)
        ax[idx].set_title(titleText)
        # title.set_fontsize(100)
    else:
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.imshow(img, interpolation='nearest');
        ax.set_title(titleText);
        # plt.show();


def show_images(show, ax_show=False):
    imshow_next.imgIdx = -1
    imshow_next.show = show
    imshow_next.ax_show = ax_show
    if ax_show:
        imshow_next.fig, imshow_next.axes = plt.subplots(ncols=2, nrows=4, figsize=(10, 5))
        imshow_next.ax = imshow_next.axes.ravel()
    else:
        plt.clf()
        plt.cla()
        plt.close()
    return show


# Conda cheat sheet: https://conda.io/docs/_downloads/conda-cheatsheet.pdf
# OpenCV 3.4 Docs: https://goo.gl/TmaFWS
#
# keywords: hsv inrange erode dilate smooth gaussian canny

# print('Load and show an image')

def generate_canny(img, do_show_images=False):
    start_time = current_milli_time()
    show_images(do_show_images, False)

    # Default mode for images in python is BGR, imshow from matplotlib uses RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imshow_next(RGB_img, "RGB")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # combine low range red thresh and high range red thresh
    # imshow_next(hsv_img, "HSV")

    # threshold on low range of HSV red
    img_thresh_low = cv2.inRange(hsv_img, np.array([0, 135, 135]), np.array([15, 255, 255]))
    # threshold on high range of HSV red
    img_thresh_high = cv2.inRange(hsv_img, np.array([159, 135, 135]), np.array([179, 255, 255]))

    # combine low range red thresh and high range red thresh
    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)
    # imshow_next(img_thresh,"Thresholded")

    # Default kernel
    kernel = np.ones((3, 3), np.uint8)

    # Erosion erodes boundaries of the foreground object, diminishes it's features: https://goo.gl/6y6DcR
    img_eroded = cv2.erode(img_thresh, kernel, iterations=1)
    imshow_next(img_eroded, "Eroded")

    # Increases the object area, accentuate image: https://goo.gl/6y6DcR
    img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
    imshow_next(img_dilated, "Dilated")

    # cv2.smoothGaussian(img_thresh, 3)
    img_gaussian_blur = cv2.GaussianBlur(img_dilated, (3, 3), sigmaX=0)
    imshow_next(img_gaussian_blur, "Gaussian Smooth")

    img_canny = cv2.Canny(img_gaussian_blur, 160, 80)
    imshow_next(img_canny, "Canny")

    # Fix spacing, make images look good
    plt.tight_layout()

    end_time = current_milli_time()
    if do_show_images:
        print("End  Time: %s" % end_time)

    if do_show_images:
        print("Elapsed: %sms" % (end_time - start_time))
    return img_canny


def get_cones(img_canny, do_show_images=False, generate_images=False):
    start_time = current_milli_time()
    if generate_images:
        show_images(do_show_images);
        img_contours = copy_blank(img_canny);
        img_all_convex_hulls = copy_blank(img_canny);
        img_all_convex_3to10_hulls = copy_blank(img_canny);
        img_traffic_cones = copy_blank(img_canny);
        img_traffic_cones_overlaps_removed = copy_blank(img_canny);

    # Find contours: https://goo.gl/FkomSt
    temp_img, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawcontours
    # cv2.drawContours(img_canny, contours, -1, (0,255,0), 3)
    # imshow_next(img_canny, ax, "Contours")

    list_traffic_cones = []
    for cnt in contours:
        # epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 3, True)
        if generate_images: cv2.drawContours(img_contours, [approx], 0, (255, 255, 100), 3)

        hull = cv2.convexHull(cnt)
        if generate_images: cv2.drawContours(img_all_convex_hulls, [hull], 0, (0, 255, 255), 3)

        if hull.size >= 10:
            # if convex hull has at least 3 and less than 10 points,
            # size returns double of actual points (2 coordinates counted for each point)
            if generate_images: cv2.drawContours(img_all_convex_3to10_hulls, [hull], 0, (0, 255, 255), 3)

            isUp, _tmp, _tmp2 = hull_pointing_up(hull);
            (x, y), (MA, ma), angle = cv2.fitEllipse(hull)
            if isUp:  # and (angle < 10):
                # if (1 == 1):
                M = cv2.moments(hull)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                font = cv2.FONT_HERSHEY_SIMPLEX
                sizelocation = (cX - 20, cY - 20)
                anglelocation = (cX - 20, cY)
                fontScale = .5
                fontColor = (255, 255, 255)
                lineType = 1
                lineType2 = cv2.LINE_AA

                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                if generate_images:
                    img_to_draw_on = img_traffic_cones
                else:
                    img_to_draw_on = img_canny

                cv2.drawContours(img_to_draw_on, [box], 0, (255, 255, 255), 2)
                cv2.drawContours(img_to_draw_on, [hull], 0, (255, 255, 255), 2)
                cv2.putText(img_to_draw_on, str(hull.size), sizelocation,
                            font, fontScale, fontColor, lineType, lineType2)
                cv2.putText(img_to_draw_on, str(angle), anglelocation,
                            font, fontScale, fontColor, lineType, lineType2)

                list_traffic_cones.append(hull)

    if generate_images:
        imshow_next(img_contours, "Contours")
        imshow_next(img_all_convex_hulls, "Hulls")
        imshow_next(img_all_convex_3to10_hulls, "Hulls3To10")
        imshow_next(img_traffic_cones, "TrafficCones")

    # list_traffic_conesWithOverlapsRemoved = removeInnerOverlappingCones(list_traffic_cones)

    # for trafficCone in list_traffic_cones:
    #    cv2.drawContour(img_traffic_cones_overlaps_removed, trafficCone, -1, (0,255,255), 2)
    #    drawGreenDotAtConeCenter(trafficCone, img_traffic_cones_overlaps_removed)

    # Fix spacing, make images look good
    if generate_images: plt.tight_layout()

    end_time = current_milli_time()
    if do_show_images:
        print("Elapsed: %sms" % (end_time - start_time))

    if generate_images:
        return list_traffic_cones, img_traffic_cones
    else:
        return list_traffic_cones, None

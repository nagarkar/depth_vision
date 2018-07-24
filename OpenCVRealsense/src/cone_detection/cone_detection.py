import numpy as np

from cone_detection.utilities import *


def copy_blank(img):
    height, width, channels = img.shape
    return np.zeros((height, width, channels), np.uint8)


# Each contour is an array of boundary points, each point enclosed in a redundant 1-size array.
# contour[0][0] is the first boundary point.
# contour[1][0] is the second boundary point.
# contour[1][0][1] is the 'y' coordinate of the second boundary point.
# Convex Hull passed in is a contour.
def hull_pointing_up(hull, min_aspect_ratio=0.8, max_top_bottom_width_ratio=0.75):
    x, y, w, h = cv2.boundingRect(hull)  # (x,y): top-left coordinate
    aspect_ratio = float(w) / h
    if aspect_ratio > min_aspect_ratio:
        return False, [], [];

    c_y = (y + h) / 2.0
    points_above_center = []
    points_below_center = []
    for point in hull:
        y = point[0][1]
        if y < c_y:
            points_above_center.append(point)
        else:
            points_below_center.append(point)
    left_most_x_below_center = hull[0][0][0];
    right_most_x_below_center = hull[0][0][0];
    for point in points_below_center:
        x = point[0][0]
        if x < left_most_x_below_center:
            left_most_x_below_center = x
        else:
            right_most_x_below_center = x

    for point in points_above_center:
        x = point[0][0]
        if x < left_most_x_below_center or x > right_most_x_below_center:
            return False, points_above_center, points_below_center

    left_most_x_above_center = hull[0][0][0]
    right_most_x_above_center = hull[0][0][0]
    for point in points_above_center:
        x = point[0][0]
        if x < left_most_x_above_center:
            left_most_x_above_center = x
        else:
            right_most_x_above_center = x
            continue

    # Find widths and avoid zero pixel widths
    top_width = max(1, abs(right_most_x_above_center - left_most_x_above_center))
    bottom_width = max(1, abs(right_most_x_below_center - left_most_x_below_center))
    shape_ratio = top_width / bottom_width
    if shape_ratio > max_top_bottom_width_ratio:
        return False, points_above_center, points_below_center

    return True, points_above_center, points_below_center

# Conda cheat sheet: https://conda.io/docs/_downloads/conda-cheatsheet.pdf
# OpenCV 3.4 Docs: https://goo.gl/TmaFWS
#
# keywords: hsv inrange erode dilate smooth gaussian canny

# print('Load and show an image')


# Assume input image is in BGR format
def generate_canny(img, do_thresholding=True, do_erode=2, do_dilate=2, do_blur=11):

    filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # combine low range red thresh and high range red thresh
    # imshow_next(hsv_img, "HSV")

    if do_thresholding:
        # threshold on low range of HSV red
        img_thresh_low = cv2.inRange(filtered_img, np.array([0, 135, 135]), np.array([15, 255, 255]))
        # img_thresh_low = cv2.inRange(hsv_img, np.array([0, 70, 50]), np.array([15, 255, 255]))
        # threshold on high range of HSV red
        img_thresh_high = cv2.inRange(filtered_img, np.array([159, 135, 135]), np.array([179, 255, 255]))
        # img_thresh_high = cv2.inRange(hsv_img, np.array([159, 70, 50]), np.array([179, 255, 255]))

        # combine low range red thresh and high range red thresh
        filtered_img = cv2.bitwise_or(img_thresh_low, img_thresh_high)
        # imshow_next(img_thresh,"Thresholded")

    # Default kernel
    kernel = np.ones((3, 3), np.uint8)

    # Erosion erodes boundaries of the foreground object, diminishes it's features: https://goo.gl/6y6DcR
    if do_erode > 0:
        filtered_img = cv2.erode(filtered_img, kernel, iterations=do_erode)
        # imshow_next(filtered_img, "Eroded")

    # Increases the object area, accentuate image: https://goo.gl/6y6DcR
    if do_dilate > 0:
        filtered_img = cv2.dilate(filtered_img, kernel, iterations=do_dilate)
        # imshow_next(filtered_img, "Dilated")

    if do_blur > 0:
        filtered_img = cv2.medianBlur(filtered_img, do_blur)
        # filtered_img = cv2.GaussianBlur(filtered_img, (do_blur, do_blur), sigmaX=0)
        # imshow_next(filtered_img, "Gaussian Smooth")

    filtered_img = cv2.Canny(filtered_img, 160, 80)
    # imshow_next(filtered_img, "Canny")

    # Fix spacing, make images look good
    # plt.tight_layout()

    return filtered_img


def get_cones(img_canny, do_polyapprox=True, base_img=None, check_countour_size=True, check_pixel_area=True):
    start_time = current_milli_time()

    # Find contours: https://goo.gl/FkomSt
    temp_img, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours
    # cv2.drawContours(img_canny, contours, -1, (0,255,0), 3)
    # imshow_next(img_canny, ax, "Contours")

    list_traffic_cones = []
    list_of_areas = []

    min_pix_area = 200
    approx_poly_dp_epsilon = 2

    for contour in contours:
        # epsilon = 0.1 * cv2.arcLength(cnt, True)
        if do_polyapprox:
            contour = cv2.approxPolyDP(contour, approx_poly_dp_epsilon, True)

        contour = cv2.convexHull(contour)

        area = cv2.contourArea(contour, False)
        list_of_areas.append(area)

        if check_countour_size and 3 < contour.size < 10:
            continue

        if check_pixel_area and area < min_pix_area:
            continue

        # if convex hull has at least 3 and less than 10 points,
        # size returns double of actual points (2 coordinates counted for each point)

        pointing_up, _tmp, _tmp2 = hull_pointing_up(contour)
        # (x, y), (MA, ma), angle = cv2.fitEllipse(hull)
        if not pointing_up:  # and (angle < 10):
            continue

        img_to_draw_on = img_canny

        cv2.drawContours(img_to_draw_on, [contour], 0, (255, 255, 255), 2)

        if contour is not None:
            list_traffic_cones.append(contour)


    # list_traffic_conesWithOverlapsRemoved = removeInnerOverlappingCones(list_traffic_cones)

    # for trafficCone in list_traffic_cones:
    #    cv2.drawContour(img_traffic_cones_overlaps_removed, trafficCone, -1, (0,255,255), 2)
    #    drawGreenDotAtConeCenter(trafficCone, img_traffic_cones_overlaps_removed)

    # Fix spacing, make images look good
    # if generate_images:
    #     plt.tight_layout()

    end_time = current_milli_time()
    return list_traffic_cones, None


def imprint_cone(cone, image):
    cv2.drawContours(image, [cone], 0, (255, 255, 255), 2)


def imprint_value(c_row, c_col, image, label, value, line_no):
    line = (c_col - 20, c_row - 10 * (line_no - 1))
    put_text_with_defaults(image, '%s:%8.3f' % (label, value), line, color=cv2_color_bgr_black,
                           font_scale=0.3)

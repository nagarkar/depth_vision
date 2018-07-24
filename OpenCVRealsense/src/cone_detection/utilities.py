import math
import os
import time

import cv2

cv2_color_bgr_red = (0, 0, 255)
cv2_color_bgr_white = (255, 255, 255)
cv2_color_bgr_black = (0, 0, 0)

_cd_def_font = cv2.FONT_HERSHEY_SIMPLEX
_cd_def_fontScale = .5
_cd_def_fontColor = (255, 255, 255)
_cd_def_lineType = 1
_cd_def_lineType2 = cv2.LINE_AA




def static_var(var_name, value):
    def decorate(func):
        setattr(func, var_name, value)
        return func
    return decorate


def current_milli_time():
    return int(round(time.clock() * 1000))


def create_dirs(dirpath):
    img_output_dir = dirpath
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)


def get_centroid(contour, width=None, height=None):
    moments = cv2.moments(contour, False)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    if width is not None:
        cy = min(cy, width - 1)
    if height is not None:
        cx = min(cx, height - 1)
    return cy, cx


@static_var("ratio_meter_to_px", None)
def get_image_area_meters(pixel_area, fov, image_depth, c_row, c_col):
    #ratio_meter_to_px = get_image_area_meters.ratio_meter_to_px
    #if ratio_meter_to_px is None:
    half_theta_wid = fov[0] / 2
    tan_half_theta_wid = math.tan(math.radians(half_theta_wid))
    # half_theta_hgt = fov[1] / 2
    half_hgt = 480 / 2
    half_wid = 640 / 2

    x = abs(half_wid - c_col)
    y = abs(half_hgt - c_row)

    hyp = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    a = half_wid / tan_half_theta_wid
    depth_pix = math.sqrt(math.pow(hyp , 2) + math.pow(a, 2))

    ratio_meter_to_px = image_depth / depth_pix
#        ratio_meter_to_px = get_image_area_meters.ratio_meter_to_px

    return pixel_area * ratio_meter_to_px * ratio_meter_to_px


def get_image_area_cm2(pixel_area, fov, image_depth, c_row, c_col):
    return get_image_area_meters(pixel_area, fov, image_depth, c_row, c_col) * 100 * 100


def put_text_with_defaults(img_to_draw_on, text, location, color=cv2_color_bgr_white, font_scale=_cd_def_fontScale):
    cv2.putText(img_to_draw_on, text, location,
                _cd_def_font,
                font_scale,
                color,
                _cd_def_lineType,
                _cd_def_lineType2)

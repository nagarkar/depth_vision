#!~/.virtualenvs/cv/bin/python
# This has been copied verbatim from https://goo.gl/t6KUWz
# Also look at https://github.com/miguelgrinberg/flask-video-streaming for alternative streaming approaches
# If this test works, you'll see a window with the color and depth images.
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from cone_detection.utilities import *
from cone_detection.cone_detection import *
from my_realsense.my_realsense import  *

# Supported values: 15, 30
fps = 15
width = 640
height = 480

# Min and max areas based on a test cone of height 20cm and width 10 cm.
min_area_cm2 = 100
max_area_cm2 = 150

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# Start streaming
pipeline.start(config)
framecount = 0
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        #time.sleep(1);
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        scale = get_depth_scale(pipeline)
        fov = get_fov(pipeline)

        list_of_cones = detect_once(color_image)

        for cone in list_of_cones:
            c_row, c_col = get_centroid(cone)
            c_row = min(c_row, height - 1)
            c_col = min(c_col, width - 1)

            image_depth = scale * depth_image[c_row, c_col]

            pixel_area = cv2.contourArea(cone)
            actual_area = get_image_area_cm2(pixel_area, fov, image_depth, c_row, c_col)
            # round(pixel_area * scale * scale * 100 * 100 * 2)

            # put_text_with_defaults(color_image, '(%s,%s)' % (cx, cy), location)
            line1 = (c_row - 20, c_col - 20)
            line2 = (c_row, c_col)

            if min_area_cm2 < actual_area < max_area_cm2 and\
                    .5 * height / 2  < c_row < 1.5 * height / 2 and\
                    .5 * width / 2  < c_col < 1.5 * width / 2:
                put_text_with_defaults(color_image, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_black,
                                       font_scale=0.3)
                put_text_with_defaults(depth_colormap, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_white,
                                       font_scale=0.3)
                put_text_with_defaults(depth_colormap, 'pa:(%s)' % pixel_area, line2, color=cv2_color_bgr_white,
                                       font_scale=0.3)

                cv2.drawContours(color_image, [cone], 0, (255, 255, 255), 2)
                cv2.drawContours(depth_colormap, [cone], 0, (255, 255, 255), 2)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        framecount = framecount + 1
finally:

    # Stop streaming
    pipeline.stop()
    print("Framecount:%s" %(framecount))

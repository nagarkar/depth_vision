#!~/.virtualenvs/cv/bin/python
# This has been copied verbatim from https://goo.gl/t6KUWz
# Also look at https://github.com/miguelgrinberg/flask-video-streaming for alternative streaming approaches
# If this test works, you'll see a window with the color and depth images.

# Usage: sudo python (3.5) <program name> ; TODO: Do this without sudo
# Note, read Readme file first.

import numpy as np
from cone_detection.utilities import *

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
framecount = 0
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
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

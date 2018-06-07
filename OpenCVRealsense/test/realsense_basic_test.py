# Simple test program to see if we can start and stop librealsense.
# Setup instructions for windows:
# Compile and install from sources as documented here: https://goo.gl/ryqSHQ
#  - I had to also install windows sdk 8.1 using the visual studio installer
#  - Make sure you pick a 64 bit visual studio option in cmake-gui

import pyrealsense2 as rs

try:
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    print('pipeline created')
    pipeline.start(config)
    print('started')
    frames = pipeline.wait_for_frames()
    print('waiting for frame')
    pipeline.stop()
    print('done')
except:
    assert False, "Real sense Basic Test Failed"

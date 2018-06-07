# API How to: https://github.com/IntelRealSense/librealsense/wiki/API-How-To

import pyrealsense2 as rs
import numpy as np
import traceback

from cone_detection.cone_detection import generate_canny, get_cones


def start_pipeline():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    return pipeline


def stop_pipeline(rs_pipeline):
    if rs_pipeline is None:
        rs_pipeline = rs.pipeline();
    rs_pipeline.stop()


def read_n_frames(pipeline, n):
    for i in range(n):
        pipeline.wait_for_frames()


def get_frames(pipeline):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise Exception()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image


def detect_once(color_image):
    try:
        rs_canny = generate_canny(color_image)
        list_of_cones, _unused = get_cones(rs_canny)
        return list_of_cones
    except:
        traceback.print_exc()

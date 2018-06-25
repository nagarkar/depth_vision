# Simple test program to see if we can start and stop librealsense.
# Setup instructions for windows:
# Compile and install from sources as documented here: https://goo.gl/ryqSHQ
#  - I had to also install windows sdk 8.1 using the visual studio installer
#  - Make sure you pick a 64 bit visual studio option in cmake-gui

import unittest

from my_realsense.my_realsense import *


class RealsenseBasicTests(unittest.TestCase):
    def test_pipeline_start_wait_for_frames_stop(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        self.assertIsNotNone(pipeline)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()

        pipeline.stop()

    def test_start_pipeline(self):
        pipeline = start_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.get_active_profile())
        stop_pipeline(pipeline)

    def test_get_depth_filter_list(self):
        filters = get_depth_filter_list()
        self.assertEqual(len(filters), 5)
        self.assertEqual(type(filters[0]), rs.decimation_filter)
        self.assertEqual(type(filters[1]), rs.disparity_transform)
        self.assertEqual(type(filters[2]), rs.spatial_filter)
        self.assertEqual(type(filters[3]), rs.temporal_filter)
        self.assertEqual(type(filters[4]), rs.disparity_transform)

        filters = get_depth_filter_list(temporal=False)
        self.assertEqual(len(filters), 4)
        filters = get_depth_filter_list(temporal=False, d2d=False)
        self.assertEqual(len(filters), 2)
        filters = get_depth_filter_list(temporal=False, d2d=False, spatial=False)
        self.assertEqual(len(filters), 1)
        filters = get_depth_filter_list(temporal=False, d2d=False, spatial=False, decimate=False)
        self.assertEqual(len(filters), 0)

    def test_get_option_data(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        self.assertIsNotNone(pipeline)
        pipeline.start(config)
        print_option_data(pipeline.get_active_profile())


if __name__ == '__main__':
    unittest.main()

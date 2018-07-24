# Simple test program to see if we can start and stop librealsense.
# Setup instructions for windows:
# Compile and install from sources as documented here: https://goo.gl/ryqSHQ
#  - I had to also install windows sdk 8.1 using the visual studio installer
#  - Make sure you pick a 64 bit visual studio option in cmake-gui

import unittest

from my_realsense.my_realsense import *


class RealsenseBasicTests(unittest.TestCase):
    pipeline = None

    def setUp(self):
        self.pipeline = start_pipeline()

    def tearDown(self):
        self.pipeline.stop()

    def test_start_stop_pipeline(self):
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.get_active_profile())

    def test_pipeline_start_wait_for_frames_stop(self):
        self.pipeline.stop()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline = rs.pipeline()
        self.assertIsNotNone(self.pipeline)
        self.pipeline.start(config)
        frames = self.pipeline.wait_for_frames()
        self.assertIsNotNone(frames)
        # self.assertIs(len(frames), 2)

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
        self.pipeline.stop()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline = rs.pipeline()
        self.assertIsNotNone(self.pipeline)
        self.pipeline.start(config)
        print_option_data(self.pipeline.get_active_profile())


if __name__ == '__main__':
    unittest.main()

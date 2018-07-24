import signal
import threading
import time

import cv2
import pyrealsense2 as rs

from my_realsense.my_realsense import *


class VideoGrabberConfig:
    record_to_file = "video.bag"
    preset_file = '../configurations/HighDensityPreset.json'
    capture_color = True
    capture_depth = True
    fps = 15
    width = 640
    height = 480

    def __init__(self, record_to_file):
        if record_to_file is not None:
            self.record_to_file = record_to_file


class VideoGrabber:
    video_grabber_config = None
    pipeline = None

    def __init__(self, video_grabber_config: VideoGrabberConfig):
        self.video_grabber_config = video_grabber_config
        signal.signal(signal.SIGTERM, self.handler)
        signal.signal(signal.SIGINT, self.handler)

    def start(self, timeout):
        self.pipeline = start_pipeline(advanced_mode=True,
                                       width=self.video_grabber_config.width,
                                       height=self.video_grabber_config.height,
                                       fps=self.video_grabber_config.fps,
                                       preset_file=self.video_grabber_config.preset_file,
                                       record_to_file=self.video_grabber_config.record_to_file)
        if timeout is not None:
            t = threading.Timer(timeout, self.stop, kwargs={'signum': None})
            t.start()

        while True:
            if self.pipeline is None:
                break
            time.sleep(1)

    def stop(self, signum):
        self.pipeline.stop()
        self.pipeline = None
        if signal is not None:
            print('Stop called as signal handler with signal %s' % signum)

    def playback(self):
        playback_window_title_suffix = 'Streaming Loop (esc to cancel)'
        depth_window = 'Depth Frames ' + playback_window_title_suffix
        color_window = 'Color Frames ' + playback_window_title_suffix
        cv2.namedWindow(depth_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(color_window, cv2.WINDOW_AUTOSIZE)
        self.pipeline = start_pipeline(advanced_mode=True,
                                       width=self.video_grabber_config.width,
                                       height=self.video_grabber_config.height,
                                       fps=self.video_grabber_config.fps,
                                       preset_file=self.video_grabber_config.preset_file,
                                       from_file=self.video_grabber_config.record_to_file)
        while True:
            if self.pipeline is None:
                break
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_frame = rs.colorizer().colorize(depth_frame)
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # images = np.hstack((color_image, depth_image))

            # Show images
            cv2.imshow(color_window, color_image)
            cv2.imshow(depth_window, depth_image)
            key = cv2.waitKey(1)
            # Exit on esc key
            if key == 27:
                cv2.destroyAllWindows()
                break

    def handler(self, signum):
        self.stop(signum)


if __name__ == '__main__':
    # Get output dir
    # output_dir = input("Output Directory:")
    config = VideoGrabberConfig("my_video.bag")
    grabber = VideoGrabber(config)
    grabber.start(1)

    grabber.playback()

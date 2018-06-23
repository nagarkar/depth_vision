from my_realsense.my_realsense import *
import cv2
import traceback
from cone_detection.utilities import *
import time
from queue import Queue
import threading

print('Initializing')

# IO and File Setup
test_output_dir = '../test-output/realsense-helper/'
create_dirs(test_output_dir)
video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 2*480))
# video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), fourcc, 20.0, (640, 480))
# video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), fourcc, 20.0, (320, 240))
# depth_video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test_depth.avi'), fourcc, 20.0, (640, 480))

# Configuration
filters_on = True
advanced_mode = False
min_area_cm2 = 50
max_area_cm2 = 200
fps = 15
seconds = 30


def abort():
    print("Aborting program")
    exit(1)


# State
frame_q = Queue()
frames_processed = 0

# Pre-processing
filters = get_depth_filter_list()
align_to = rs.align(rs.stream.depth)

pipeline = start_pipeline(advanced_mode, fps)
if not pipeline:
    abort()

scale = get_depth_scale(pipeline)
fov = get_fov(pipeline)

print('completed initialization')
print('runtime expected: %s' % seconds)

# Read one second worth of frames to stabilize frames
read_n_frames(pipeline, 3 * fps)


def process_frames():
    global frames_processed

    while True:
        depth_frame, color_frame = frame_q.get()

        if filters_on:
            depth_frame = apply_filters(filters, depth_frame)

        depth_image, color_image = get_frames_from_raw(depth_frame, color_frame)
        if filters_on:
            depth_image = cv2.resize(depth_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        list_of_cones = detect_once(color_image)

        for cone in list_of_cones:
            c_row, c_col = get_centroid(cone)
            c_row = min(c_row, 480 - 1)
            c_col = min(c_col, 640 - 1)

            image_depth = scale * depth_image[c_row, c_col]

            pixel_area = cv2.contourArea(cone)
            actual_area = get_image_area_cm2(pixel_area, fov, image_depth, c_row, c_col)
            # round(pixel_area * scale * scale * 100 * 100 * 2)

            # put_text_with_defaults(color_image, '(%s,%s)' % (cx, cy), location)
            line1 = (c_row - 20, c_col - 20)
            line2 = (c_row, c_col)

            if min_area_cm2 < actual_area < max_area_cm2:
                put_text_with_defaults(color_image, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_black,
                                       font_scale=0.3)
                put_text_with_defaults(depth_colormap, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_white,
                                       font_scale=0.3)
                put_text_with_defaults(depth_colormap, 'pa:(%s)' % pixel_area, line2, color=cv2_color_bgr_white,
                                       font_scale=0.3)

                cv2.drawContours(color_image, [cone], 0, (255, 255, 255), 2)
                cv2.drawContours(depth_colormap, [cone], 0, (255, 255, 255), 2)

        # cv2.drawContours(color_image, list_of_cones, 0, (255, 255, 255), 2)
        # cv2.drawContours(depth_colormap, list_of_cones, 0, (255, 255, 255), 2)

        images = np.vstack((color_image, depth_colormap))
        video_out.write(images)
        # video_out.write(color_image)
        # video_out.write(depth_colormap)

        #       depth_video_out.write(depth_colormap)
        frames_processed = frames_processed + 1
        frame_q.task_done()


t = threading.Thread(target=process_frames)
t.daemon = True
t.start()


def start_stream():

    try:
        start_time = time.clock()
        for i in range(fps * seconds):

            depth_frame, color_frame = get_raw_frames(pipeline, align_to)
            frame_q.put((depth_frame, color_frame))

        print('done queueing frames in %s seconds' % (time.clock() - start_time))
        frame_q.join()
        print('done processing frames in %s seconds' % (time.clock() - start_time))
        print('Number of frames processed: %s' % frames_processed)
        print('expected to be done in seconds: %s' % seconds)

        video_out.release()
        # depth_video_out.release()
    except:
        print('exception')
        traceback.print_exc()
        video_out.release()
        # depth_video_out.release()
        pipeline.stop()


start_stream()

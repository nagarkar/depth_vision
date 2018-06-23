from my_realsense.my_realsense import *
import traceback
from cone_detection.utilities import *
import time
import multiprocessing


def abort():
    print("Aborting program")
    exit(1)


# Configuration
filters_on = False
advanced_mode = False

# Min and max areas based on a test cone of height 20cm and width 10 cm.
min_area_cm2 = 80
max_area_cm2 = 150


# Supported values: 15, 30
fps = 15
width = 640
height = 480

seconds = 5
throwaway_seconds = 5


def info(title):
    print(title)
    # print('module name:', __name__)
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())


def frames_to_images(lock, image_q):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # pipeline = start_pipeline(advanced_mode, fps, width, height)
    # if not pipeline:
    #     abort()

    scale = get_depth_scale(pipeline)
    fov = get_fov(pipeline)

    # Read one second worth of frames to stabilize frames
    read_n_frames(pipeline, throwaway_seconds * fps)

    info('frames_to_images')
    # Pre-processing
    filters = get_depth_filter_list()
    n_frames = 1
    print("Started Collecting Frames at time : %s" % time.time())
    try:
        # align_to = rs.align(rs.stream.depth)
        for i in range(int(fps * seconds)):
            depth_frame, color_frame = get_raw_frames(pipeline)
            with lock:
                # print("processing frame %s" % n_frames)
                n_frames = n_frames + 1

            if filters_on:
                depth_frame = apply_filters(filters, depth_frame)

            depth_image, color_image = get_frames_from_raw(depth_frame, color_frame)

            image_q.put((color_image, depth_image, scale, fov))
    except:
        with lock:
            traceback.print_exc()
            print("exception in process_frames")
    finally:
        pipeline.stop()


def images_to_video(lock, image_q):
    global fps, width, height

    info('images_to_video')

    test_output_dir = '../test-output/realsense-helper/'
    create_dirs(test_output_dir)
    video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), cv2.VideoWriter_fourcc(*'XVID')
                                  , 20.0, (width, 2 * height))
    try:
        n_images = 1
        for i in range(int(fps * seconds)):
            # print("processing image %s" % n_images)
            n_images = n_images + 1

            color_image, depth_image, scale, fov = image_q.get()

            if filters_on:
                depth_image = cv2.resize(depth_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

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

                if min_area_cm2 < actual_area < max_area_cm2:
                    put_text_with_defaults(color_image, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_black,
                                           font_scale=0.3)
                    put_text_with_defaults(depth_colormap, 'aa:(%s)' % actual_area, line1, color=cv2_color_bgr_white,
                                           font_scale=0.3)
                    put_text_with_defaults(depth_colormap, 'pa:(%s)' % pixel_area, line2, color=cv2_color_bgr_white,
                                           font_scale=0.3)

                    cv2.drawContours(color_image, [cone], 0, (255, 255, 255), 2)
                    cv2.drawContours(depth_colormap, [cone], 0, (255, 255, 255), 2)

            images = np.vstack((color_image, depth_colormap))
            video_out.write(images)
    except:
        with lock:
            traceback.print_exc()
            print('Exception in write_out_frames')
    finally:
        video_out.release()


if __name__ == '__main__':

    g_lock = multiprocessing.Lock()
    print('Initializing')

    try:

        # @atexit.register
        # def cleanup():
        #     pipeline.stop()
        #     g_video_out.release()

        print('completed initialization')
        print('runtime expected: %s seconds' % seconds)

        g_image_q = multiprocessing.Queue()
        frames_to_images_proc = multiprocessing.Process(target=frames_to_images, args=(g_lock, g_image_q))
        frames_to_images_proc.start()

        images_to_video_proc = multiprocessing.Process(target=images_to_video, args=(g_lock, g_image_q))
        images_to_video_proc.start()

        frames_to_images_proc.join()
        with g_lock:
            print('done queueing frames in %s seconds' % time.time())

        images_to_video_proc.join()
        with g_lock:
            print('done processing frames in %s seconds' % time.time())

    except:
        print('exception in start_stream')
        traceback.print_exc()


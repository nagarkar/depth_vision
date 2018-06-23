from my_realsense.my_realsense import *
import traceback
from cone_detection.utilities import *
import time
import multiprocessing
import atexit

def abort():
    print("Aborting program")
    exit(1)


# Configuration
filters_on = True
advanced_mode = False
min_area_cm2 = 50
max_area_cm2 = 200
fps = 15
seconds = .2
throwaway_seconds = 0


def info(title):
    print(title)
    # print('module name:', __name__)
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)


def frames_to_images(lock, frame_q, image_q):
    filters = get_depth_filter_list()
    try:
        n_frames = 1
        for i in range(int(fps * seconds)):
            depth_frame, color_frame, scale, fov = frame_q.get()
            with lock:
                print("processing frame to image %s" % n_frames)
                n_frames = n_frames + 1

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

            image_q.put((color_image, depth_colormap))
    except:
        with lock:
            traceback.print_exc()
            print("exception in process_frames")


def images_to_video(lock, image_q):
    global fps

    info('images_to_video')

    test_output_dir = '../test-output/realsense-helper/'
    create_dirs(test_output_dir)
    video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), cv2.VideoWriter_fourcc(*'XVID')
                                  , 20.0, (640, 2 * 480))
    # video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), fourcc, 20.0, (640, 480))
    # video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), fourcc, 20.0, (320, 240))
    # depth_video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test_depth.avi'), fourcc, 20.0, (640, 480))
    try:
        n_images = 1
        for i in range(int(fps * seconds)):
            print("processing image %s" % n_images)
            n_images = n_images + 1

            color_image, depth_colormap = image_q.get()
            images = np.vstack((color_image, depth_colormap))
            video_out.write(images)
    except:
        with lock:
            traceback.print_exc()
            print('Exception in write_out_frames')
    finally:
        video_out.release()


def pipeline_to_frames(lock, frames_q):

    align_to = rs.align(rs.stream.depth)
    pipeline = start_pipeline(advanced_mode, fps)
    if not pipeline:
        abort()

    scale = get_depth_scale(pipeline)
    fov = get_fov(pipeline)

    # Read one second worth of frames to stabilize frames
    read_n_frames(pipeline, throwaway_seconds * fps)

    info('frames_to_images')
    n_frames = 1
    try:
        for i in range(int(fps * seconds)):
            depth_frame, color_frame = get_raw_frames(pipeline, align_to)
            frames_q.put((depth_frame, color_frame, scale, fov))
            with lock:
                print("processing frame %s" % n_frames)
                n_frames = n_frames + 1

    except:
        with lock:
            traceback.print_exc()
            print("exception in pipeline to frames")
    finally:
        pipeline.stop()


# def start_stream(frame_q, pipeline, align_to):


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

        g_frame_q = multiprocessing.Queue()
        g_image_q = multiprocessing.Queue()
        pipeline_to_frames_proc = multiprocessing.Process(target=pipeline_to_frames, args=(g_lock, g_frame_q))
        pipeline_to_frames_proc.start()

        frames_to_images_proc = multiprocessing.Process(target=frames_to_images, args=(g_lock, g_frame_q, g_image_q))
        frames_to_images_proc.start()

        images_to_video_proc = multiprocessing.Process(target=images_to_video, args=(g_lock, g_image_q))
        images_to_video_proc.start()

        start_time = time.clock()

        pipeline_to_frames_proc.join()
        frames_to_images_proc.join()
        images_to_video_proc.join()
        # while not g_image_q.empty():
        #     time.sleep(1/fps)

        with g_lock:
            print('done queueing frames in %s seconds' % (time.clock() - start_time))

        with g_lock:
            print('done processing frames in %s seconds' % (time.clock() - start_time))
            print('expected to be done in seconds: %s' % seconds)
    except:
        print('exception in start_stream')
        traceback.print_exc()


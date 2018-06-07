from my_realsense.my_realsense import *
import cv2
import traceback
from cone_detection.utilities import *

print('starting')
test_output_dir = '../test-output/realsense-helper/'
create_dirs(test_output_dir)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test.avi'), fourcc, 20.0, (640, 2*480))
#depth_video_out = cv2.VideoWriter(os.path.join(test_output_dir, 'video-test_depth.avi'), fourcc, 20.0, (640, 480))

try:
    pipeline = start_pipeline()
    print('started')

    # Stabilize frames
    read_n_frames(pipeline, 30);

    fps = 30
    seconds = 1
    for i in range(fps*seconds):
        depth_image, color_image = get_frames(pipeline)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        list_of_cones = detect_once(color_image)

        cv2.drawContours(color_image, list_of_cones, 0, (255, 255, 255), 2)
        cv2.drawContours(depth_colormap, list_of_cones, 0, (255, 255, 255), 2)

        images = np.vstack((color_image, depth_colormap))

        video_out.write(images)
 #       depth_video_out.write(depth_colormap)
    print('done')
    video_out.release()
  #  depth_video_out.release()
except:
    print('exception')
    traceback.print_exc()
    video_out.release()
    #depth_video_out.release()
    pipeline.stop()

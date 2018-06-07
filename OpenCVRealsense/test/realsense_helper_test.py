from my_realsense.my_realsense import *
import cv2
import traceback
from cone_detection.utilities import *

print('starting')
img_output_dir = '../test-output/realsense-helper/'

try:
    pipeline = start_pipeline()
    print('started')
    create_dirs(img_output_dir)

    # Stabilize frames
    read_n_frames(pipeline, 30);

    depth_image, color_image = get_frames(pipeline)
    print('got frames')
    list_of_cones = detect_once(color_image)
    cv2.drawContours(color_image, list_of_cones, 0, (255, 255, 255), 2)
    # print('drew contours')

    cv2.imwrite(os.path.join(img_output_dir, 'color_image_detected_cones.jpg'), color_image)
    print('done')
except:
    print('exception')
    traceback.print_exc()
    pipeline.stop()

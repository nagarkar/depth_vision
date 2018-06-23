from cone_detection.cone_detection import *
import unittest


class TestSingleConeDetection(unittest.TestCase):
    def test_basic_cone_detection(self):
        img_input_dir = '../images'
        img_output_dir = '../detections'
        filename = "15.png"

        img = cv2.imread(os.path.join(img_input_dir, filename))
        img_canny = generate_canny(img, True)
        list_of_cones, _unused = get_cones(img_canny, img, True, True)
        # notice -1 for "Write all contours to image"
        cv2.drawContours(img, list_of_cones, -1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(img_output_dir, 'canny_' + filename), img_canny)
        cv2.imwrite(os.path.join(img_output_dir, filename), img)
        self.assertGreater(len(list_of_cones), 0, "Did not find cone in image %s" % filename)


if __name__ == '__main__':
    unittest.main()


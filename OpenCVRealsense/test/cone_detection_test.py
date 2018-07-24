import unittest

from cone_detection.cone_detection import *


class TestConeDetection(unittest.TestCase):
    expected_number_of_cones = {
        "1.jpg": 1,
        "2.jpg": 1,
        "3.jpg": 1,
        "4.jpg": 7,
        "5.jpg": 1,
        "6.png": 2,
        "7.jpg": 3,
        "8.jpg": 4,
        "9.jpg": 1,
        "10.jpg": 4,
        "11.jpg": 6,
        "12.jpg": 1,
        "13.jpg": 1,
        "14.jpg": 1,  # TODO: this should detect 1
        "15.png": 1,
    }

    def verify(self, img_input_dir, img_output_dir, filename, expected_num_cones):
        img = cv2.imread(os.path.join(img_input_dir, filename))
        img_canny = generate_canny(img)
        list_of_cones, _unused = get_cones(img_canny)
        # notice -1 for "Write all contours to image"
        cv2.drawContours(img, list_of_cones, -1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(img_output_dir, 'canny_' + filename), img_canny)
        cv2.imwrite(os.path.join(img_output_dir, filename), img)
        self.assertGreater(len(list_of_cones), 0, "Did not find cone in image %s" % filename)
        self.assertEqual(expected_num_cones, len(list_of_cones),
                         'Unexpected number of cones in %s' % filename)

    def test_basic(self):
        test_contour = np.array([[[885, 375]], [[885, 377]], [[890, 377]], [[889, 376]], [[890, 375]]])
        self.assertEqual(test_contour[1][0][1], 377)
        test, points_above_center, points_below_center = hull_pointing_up(test_contour)
        self.assertFalse(test, "Test Contour Failed")
        self.assertEqual(len(points_above_center), 0, "Test Contour Failed")
        self.assertEqual(len(points_below_center), 0, "Test Contour Failed")

    def test_basic_cone_detection(self):
        img_input_dir = '../images'
        img_output_dir = '../detections'
        for filename in os.listdir(img_output_dir):
            os.remove(os.path.join(img_output_dir, filename))

        for filename in os.listdir(img_input_dir):
            self.verify(img_input_dir, img_output_dir, filename, self.expected_number_of_cones[filename])


if __name__ == '__main__':
    unittest.main()

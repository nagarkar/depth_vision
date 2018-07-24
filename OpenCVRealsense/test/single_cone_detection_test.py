from cone_detection_test import *


class TestSingleConeDetection(TestConeDetection):
    def test_basic_cone_detection(self):
        img_input_dir = '../images'
        img_output_dir = '../detections'
        filename = "14.jpg"
        self.verify(img_input_dir, img_output_dir, filename, 1)


if __name__ == '__main__':
    unittest.main()


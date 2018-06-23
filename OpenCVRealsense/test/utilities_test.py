import cone_detection.utilities as util
import unittest


class TestStringMethods(unittest.TestCase):

    def test_basic(self):
        pixel_area = 1000
        fov = [53, 40]  # degrees
        image_depth = 3  # meters
        c_row = 200
        c_col = 200
        expected_area_cm2 = 200
        cm2_to_m2_factor = 100 * 100
        self.assertAlmostEqual(util.get_image_area_meters(pixel_area, fov, image_depth, c_row, c_col),
                               expected_area_cm2 / cm2_to_m2_factor, 2)


if __name__ == '__main__':
    unittest.main()


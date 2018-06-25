import unittest

from cone_detection.cone_detection import *


class TestHistogramEqualization(unittest.TestCase):
    def test_basic_cone_detection(self):
        img_input_dir = '../images'
        img_output_dir = '../detections'
        filename = "15.png"

        orig_bgr = cv2.imread(os.path.join(img_input_dir, filename))
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        plt.show()

        plt.title('Original Canny')
        plt.imshow(generate_canny(orig_bgr))
        plt.show()

        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()

        orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        # img = cv2.equalizeHist(img)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY))
        plt.title('Clahe Img')
        plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))
        plt.show()

        img = cv2.scaleAdd(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB), 0.8, orig_rgb)
        plt.title('0.8 * Clahe + Original')
        plt.imshow(img)
        plt.show()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        canny = generate_canny(img)
        plt.title('Canny on (Clahe + Original)')
        plt.imshow(canny)
        plt.show()

        cones, _unused = get_cones(canny)
        height, width, channels = img.shape

        for cone in cones:
            if cone is None:
                continue
            c_row, c_col = get_centroid(cone, width, height)
            pixel_area = cv2.contourArea(cone)

            # Assumptions
            image_depth = 2
            fov = [50, 50]

            actual_area = get_image_area_cm2(pixel_area, fov, image_depth, c_row, c_col)

            imprint_value(c_row, c_col, img, 'aa', actual_area, 1)
            imprint_value(c_row, c_col, img, 'pa', pixel_area, 2)
            imprint_cone(cone, img)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    unittest.main()

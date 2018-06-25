import unittest

from cone_detection.cone_detection import *


class TestShadowRemoval(unittest.TestCase):
    def test_basic_cone_detection(self):
        img_input_dir = '../images'
        img_output_dir = '../detections'
        filename = "15.png"

        orig_bgr = cv2.imread(os.path.join(img_input_dir, filename))
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        plt.show()

        canny = generate_canny(orig_bgr)
        plt.title('Orig Canny')
        plt.imshow(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
        plt.show()

        rgb_planes = cv2.split(orig_bgr)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = diff_img.copy()
            cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        norm_img = cv2.merge(result_norm_planes)
        # plt.title('Normalized')
        # plt.imshow(norm_img)
        # plt.show()

        masked = cv2.bitwise_and(norm_img, orig_bgr)
        # plt.title('Masked')
        # plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        # plt.show()

        masked_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        sv_limit = 100
        img_thresh_low = cv2.inRange(masked_hsv, np.array([0, sv_limit, sv_limit]), np.array([15, 255, 255]))
        img_thresh_high = cv2.inRange(masked_hsv, np.array([159, sv_limit, sv_limit]), np.array([179, 255, 255]))
        thresholded = cv2.cvtColor(cv2.bitwise_or(img_thresh_low, img_thresh_high), cv2.COLOR_GRAY2BGR)
        # plt.title('Masked + Thresholded')
        # plt.imshow(cv2.cvtColor(thresholded, cv2.COLOR_HSV2RGB))
        # plt.show()

        blurred = cv2.medianBlur(thresholded, 21)
        # plt.title('Masked + Thresholded + Blurred')
        # plt.imshow(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
        # plt.show()

        canny = cv2.Canny(blurred, 160, 80)
        # canny = generate_canny(masked_img, do_thresholding=True, do_erode=0, do_dilate=0, do_blur=11)
        plt.title('Processed Canny on mask')
        plt.imshow(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
        plt.show()

        img = cv2.imread(os.path.join(img_input_dir, filename))
        cones, _unused = get_cones(canny, base_img=orig_bgr, do_show_images=True, generate_images=True,
                                   check_countour_size=False, check_pixel_area=False)
        height, width, channels = orig_bgr.shape

        print("Found %s cones" % len(cones))
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
            cv2.drawContours(img, [cone], 0, (255, 255, 255), 2)
            # imprint_cone(cone, img)

        plt.title('Img with imprinted cones')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == '__main__':
    unittest.main()

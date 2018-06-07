import os

from cone_detection.cone_detection import *

# Test
test_contour = np.array([[[885, 375]], [[885, 377]], [[890, 377]], [[889, 376]], [[890, 375]]])
assert (test_contour[1][0][1] == 377)
test, pointsAboveCenter, pointsBelowCenter = hull_pointing_up(test_contour)
assert test is False, "Test Contour Failed"
assert len(pointsAboveCenter) == 0, "Test Contour Failed"
assert len(pointsBelowCenter) == 0, "Test Contour Failed"

test_contour = np.array([[[885, 375]], [[885, 377]], [[890, 377]], [[889, 376]], [[890, 375]]])
test, pointsAboveCenter, pointsBelowCenter = hull_pointing_up(test_contour)
assert test is False, "Test Contour Failed"
assert len(pointsAboveCenter) == 0, "Test Contour Failed"
assert len(pointsBelowCenter) == 0, "Test Contour Failed"

img_input_dir = '../images'
img_output_dir = '../detections'
for filename in os.listdir(img_input_dir):
    img = cv2.imread(os.path.join(img_input_dir, filename))
    img_canny = generate_canny(img)
    listOfCones, _unused = get_cones(img_canny, False, False)
    # notice -1 for "Write all contours to image"
    cv2.drawContours(img, listOfCones, -1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(img_output_dir, 'canny_' + filename), img_canny)
    cv2.imwrite(os.path.join(img_output_dir, filename), img)

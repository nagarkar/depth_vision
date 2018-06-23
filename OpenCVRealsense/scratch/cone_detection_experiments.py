import os

from cone_detection.cone_detection import *

img_input_dir = '../images'
img_output_dir = '../experiments'

filename = "13.jpg"

img = cv2.imread(os.path.join(img_input_dir, filename))
img_canny = generate_canny(img, True)
cv2.imwrite(os.path.join(img_output_dir, 'canny_' + filename), img_canny)
temp_img, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

idx = 0
for contour in contours:
    cnt_img = img.copy()
    cv2.drawContours(cnt_img, [contours[idx]], -1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(img_output_dir, '%s_contour_%s.jpg' %(filename, idx)), cnt_img)
    idx = idx + 1

hulls = []
for contour in contours:
    hull = cv2.convexHull(contour, returnPoints=True)
    hulls.append(hull)

areas = []
for contour in hulls:
    area = cv2.contourArea(contour)
    areas.append(area)

print(areas)

idx = 0
for contour in hulls:
    cnt_img = img.copy()
    cv2.drawContours(cnt_img, [contours[idx]], -1, (255, 255, 255), cv2.FILLED)
    cv2.imwrite(os.path.join(img_output_dir, '%s_fill_contour_%s.jpg' % (filename, idx)), cnt_img)
    idx = idx + 1

# idx = 0
# for contour in contours:
#     cnt_img = img.copy()
#     cv2.fillPoly(cnt_img, contour, (255, 255, 255))
#     cv2.imwrite(os.path.join(img_output_dir, '%s_fill_contour_%s.jpg' %(filename, idx)), cnt_img)
#     idx = idx + 1


# listOfCones, _unused = get_cones(img_canny, img, True, True)
# # notice -1 for "Write all contours to image"
# cv2.drawContours(img, listOfCones, -1, (255, 255, 255), 2)
# cv2.imwrite(os.path.join(img_output_dir, 'canny_' + filename), img_canny)
# cv2.imwrite(os.path.join(img_output_dir, filename), img)

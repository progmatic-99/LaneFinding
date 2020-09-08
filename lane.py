"""
- Edge detection
- Gradient: Sharp Change in intensity of adjacent pixels
- Edge: Rapid change in gradient
"""

import cv2
import numpy as np

"""
- Reduce Noise
- Image Noise can create false edges
- Applying Gaussian Blur to reduce noise
- Kernel Convolution
- Strong gradient: Sharp change in adjancent pxiels
- Small gradient: Shallow change in adjancent pxiels
- Canny function
"""


def canny(image):
    # Processing a single channel is less compute expensive
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)

    return masked_img


img = cv2.imread("test_image.jpg")
lane_img = np.copy(img)
canny = canny(lane_img)
cropped_img = region_of_interest(canny)
cv2.imshow("Canny", cropped_img)
cv2.waitKey(0)

# plt.imshow(canny)
# plt.show()

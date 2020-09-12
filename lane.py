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


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, y1), (x2, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def canny(image):
    # Processing a single channel is less compute expensive
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny


def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_img


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)

    return masked_img


"""
Parametric Space
Hough space: (slope, y-intercept):(x, y)
Instead of a line, single point is used
Cartesian space: family of lines pass through a point
Split hough space into a grid
Point of intersection in a single bin, voting on grids
Bin with most votes, gives point (m, b) which corresponds to line of best fit

- Problem: Vertical lines with infinte slope can't be represented
Instead of cartesian co-ordiantes, we can take polar co-ordinates
r = x cos(n) + y sin(n) :n -> angle
Line of best fit can be detected by intersection of curves
"""

img = cv2.imread("test_image.jpg")
lane_img = np.copy(img)
canny_img = canny(lane_img)
cropped_img = region_of_interest(canny_img)
lines = cv2.HoughLinesP(
    cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
)
averaged_lines = average_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, averaged_lines)
combo = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
cv2.imshow("Result", combo)
cv2.waitKey(0)

# plt.imshow(canny)
# plt.show()

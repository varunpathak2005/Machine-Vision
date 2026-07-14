# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Load image
image_path = r"D:\Machine-Vision\Metal_Defect_Detection\002.jpg"

image = cv2.imread(image_path)

if image is None:
    raise Exception(f"Unable to read image: {image_path}")

# Resize if image is too large
max_width = 900

if image.shape[1] > max_width:
    ratio = max_width / image.shape[1]
    image = cv2.resize(
        image,
        (int(image.shape[1] * ratio),
         int(image.shape[0] * ratio))
    )

print("Original Image")
cv2.imshow("Original Image", image)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

print("Grayscale")
cv2.imshow("Grayscale", gray)

# Edge Detection
edges = cv2.Canny(gray, 50, 150)

print("Edges")
cv2.imshow("Edges", edges)

# Threshold
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)

print("Threshold")
cv2.imshow("Threshold", thresh)

# Morphology
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel
)

print("Morphology")
cv2.imshow("Morphology", closing)

# Find contours
contours, _ = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

count = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 100:

        count += 1

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Area:{int(area)}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

cv2.putText(
    output,
    f"Scratches: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

print(f"Total Scratches Detected: {count}")

cv2.imshow("Detected Scratches", output)

cv2.imwrite("processed_result.png", output)

print("Result saved as processed_result.png")

cv2.waitKey(0)
cv2.destroyAllWindows()

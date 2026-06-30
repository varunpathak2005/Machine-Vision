import cv2
import numpy as np
import os

# ==========================================
# LOAD IMAGE
# ==========================================
IMAGE_NAME = "img.jpg"

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, IMAGE_NAME)

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Image not found!")
    print("Files in folder:", os.listdir(folder))
    exit()

result = img.copy()

# ==========================================
# SCALING
# ==========================================
scaled = cv2.resize(img, None, fx=0.7, fy=0.7)

# ==========================================
# ROTATION
# ==========================================
(h, w) = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D(
    (w // 2, h // 2),
    45,
    1.0
)

rotated = cv2.warpAffine(
    img,
    rotation_matrix,
    (w, h),
    borderValue=(0, 0, 0)
)

# ==========================================
# AFFINE TRANSFORMATION
# ==========================================
pts1 = np.float32([
    [50, 50],
    [200, 50],
    [50, 200]
])

pts2 = np.float32([
    [80, 80],
    [250, 50],
    [100, 250]
])

affine_matrix = cv2.getAffineTransform(
    pts1,
    pts2
)

affine = cv2.warpAffine(
    img,
    affine_matrix,
    (w, h)
)

# ==========================================
# BETTER OBJECT DETECTION
# ==========================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edges = cv2.Canny(
    blur,
    50,
    150
)

kernel = np.ones((7, 7), np.uint8)

edges = cv2.dilate(
    edges,
    kernel,
    iterations=2
)

edges = cv2.erode(
    edges,
    kernel,
    iterations=1
)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) > 0:

    largest = max(
        contours,
        key=cv2.contourArea
    )

    # Rotated rectangle
    rect = cv2.minAreaRect(largest)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(
        result,
        [box],
        0,
        (0, 255, 0),
        3
    )

    width = int(rect[1][0])
    height = int(rect[1][1])

    print("\n===== PRODUCT DIMENSIONS =====")
    print("Width :", width, "pixels")
    print("Height:", height, "pixels")

else:
    width = 0
    height = 0
    print("No object detected!")

# ==========================================
# RESIZE FOR DISPLAY
# ==========================================
result_display = cv2.resize(result, (500, 350))
scaled_display = cv2.resize(scaled, (500, 350))
rotated_display = cv2.resize(rotated, (500, 350))
affine_display = cv2.resize(affine, (500, 350))

# ==========================================
# MEASUREMENT BOX
# ==========================================
cv2.rectangle(
    result_display,
    (330, 10),
    (495, 80),
    (255, 255, 255),
    -1
)

cv2.rectangle(
    result_display,
    (330, 10),
    (495, 80),
    (0, 0, 0),
    2
)

cv2.putText(
    result_display,
    f"W: {width}px",
    (340, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)

cv2.putText(
    result_display,
    f"H: {height}px",
    (340, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)

# ==========================================
# LABELS
# ==========================================
cv2.putText(
    result_display,
    "Measurement",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    scaled_display,
    "Scaling",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    rotated_display,
    "Rotation",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    affine_display,
    "Affine Transformation",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

# ==========================================
# COMBINE OUTPUTS
# ==========================================
top_row = np.hstack((result_display, scaled_display))
bottom_row = np.hstack((rotated_display, affine_display))

final_output = np.vstack((top_row, bottom_row))

# ==========================================
# SHOW RESULT
# ==========================================
cv2.imshow(
    "Machine Vision - Product Dimension Measurement",
    final_output
)

cv2.waitKey(0)
cv2.destroyAllWindows()
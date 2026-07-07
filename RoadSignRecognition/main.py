import cv2
import os

# ==========================================================
# Project Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "database")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test", "stop.jpg")

print("Database :", DATABASE_PATH)
print("Test Image :", TEST_IMAGE_PATH)

# ==========================================================
# Check Paths
# ==========================================================
if not os.path.exists(DATABASE_PATH):
    print("Database folder not found!")
    exit()

if not os.path.exists(TEST_IMAGE_PATH):
    print("Test image not found!")
    exit()

# ==========================================================
# Read Test Image
# ==========================================================
test_img = cv2.imread(TEST_IMAGE_PATH)

if test_img is None:
    print("Unable to load test image.")
    exit()

gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# ==========================================================
# ORB Detector
# ORB = FAST + BRIEF
# ==========================================================
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8
)

# Detect keypoints and descriptors
kp_test, des_test = orb.detectAndCompute(gray_test, None)

if des_test is None:
    print("No ORB features detected in test image.")
    exit()

# ==========================================================
# Draw ORB Keypoints
# ==========================================================
keypoint_img = cv2.drawKeypoints(
    test_img,
    kp_test,
    None,
    color=(0,255,0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# ==========================================================
# BF Matcher
# ==========================================================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

best_name = ""
best_score = 0
best_matches = None
best_db_img = None
best_kp = None

print("\nMatching Results")
print("-"*40)

# ==========================================================
# Compare with every database image
# ==========================================================
for file in os.listdir(DATABASE_PATH):

    if not file.lower().endswith((".jpg",".jpeg",".png",".bmp")):
        continue

    image_path = os.path.join(DATABASE_PATH,file)

    db_img = cv2.imread(image_path)

    if db_img is None:
        continue

    gray_db = cv2.cvtColor(db_img,cv2.COLOR_BGR2GRAY)

    kp_db, des_db = orb.detectAndCompute(gray_db,None)

    if des_db is None:
        continue

    matches = bf.match(des_db,des_test)

    matches = sorted(matches,key=lambda x:x.distance)

    good_matches = []

    for m in matches:
        if m.distance < 50:
            good_matches.append(m)

    print(f"{file:20s}  Good Matches : {len(good_matches)}")

    if len(good_matches) > best_score:
        best_score = len(good_matches)
        best_name = file
        best_matches = good_matches
        best_db_img = db_img
        best_kp = kp_db

print("-"*40)

if best_score == 0:
    print("No matching sign found.")
    cv2.imshow("Test Image",test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

print("\nDetected Road Sign :",best_name)
print("Matching Score :",best_score)

# ==========================================================
# Display Result
# ==========================================================
result = test_img.copy()

cv2.putText(
    result,
    "Detected : " + best_name,
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

# ==========================================================
# Draw Feature Matches
# ==========================================================
matched = cv2.drawMatches(
    best_db_img,
    best_kp,
    test_img,
    kp_test,
    best_matches[:40],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ==========================================================
# Show Windows
# ==========================================================
cv2.imshow("Test Image",test_img)
cv2.imshow("ORB Keypoints (FAST + BRIEF)",keypoint_img)
cv2.imshow("Detected Road Sign",result)
cv2.imshow("Feature Matching",matched)

cv2.waitKey(0)
cv2.destroyAllWindows()
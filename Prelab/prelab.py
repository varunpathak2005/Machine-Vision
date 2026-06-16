# IMAGE LOADING AND DISPLAY
import cv2

print("Program started")

img = cv2.imread("sample.jpg")

print("Image loaded:", img is not None)

if img is None:
    print("ERROR: Image not found")
else:
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Program ended")



# IMAGE PROPERTIES
import cv2

img = cv2.imread("sample.jpg")

print("Shape:", img.shape)
print("Height:", img.shape[0])
print("Width:", img.shape[1])
print("Channels:", img.shape[2])



# COLOR CHANNELS
import cv2

img = cv2.imread("sample.jpg")

blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]

cv2.imshow("Blue Channel", blue)
cv2.imshow("Green Channel", green)
cv2.imshow("Red Channel", red)

cv2.waitKey(0)
cv2.destroyAllWindows()


# WEBCAM ACCESS
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# VIDEO CAPTURE FROM FILE
import cv2

cap = cv2.VideoCapture("sample.mp4")

if not cap.isOpened():
    print("Video file not found!")
else:
    print("Video opened successfully")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
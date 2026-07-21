"""
Face and Eye Detection System using Haar Cascade Classifiers
Automated Attendance System - Prac Implementation
(Single all-in-one script)

What it does:
1. Loads an input image.
2. Detects face(s) using haarcascade_frontalface_default.xml.
3. Detects eyes inside each detected face using haarcascade_eye.xml.
4. Draws boxes (blue = face, green = eyes) and saves the annotated image + JSON report.
5. Simulates 8 different lighting conditions (bright, dark, low-contrast,
   high-contrast, gamma indoor/daylight, noisy low-light) from the SAME image,
   runs detection on each, and saves:
     - an annotated image per condition
     - a CSV summary table (brightness vs faces/eyes detected)
     - a bar chart comparing accuracy across conditions

Usage:
    python3 attendance_face_eye_detection.py <input_image_path> [output_dir]

Example:
    python3 attendance_face_eye_detection.py photo.jpg results
"""

import cv2
import numpy as np
import sys
import os
import csv
import json

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Haar Cascade classifiers bundled with OpenCV
# ---------------------------------------------------------------------------
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------
def detect_faces_and_eyes(image_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)):
    """Detect faces, then detect eyes within each face ROI. Returns annotated image + results dict."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # helps normalize uneven lighting

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )

    annotated = image_bgr.copy()
    results = {"num_faces": int(len(faces)), "faces": []}

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue face box

        # Eyes are always in the upper ~60% of the face box — restricting the
        # search region avoids the nose/mouth confusing the detector, and lets
        # us use a lower minNeighbors (more lenient) without extra false hits.
        upper_h = int(h * 0.6)
        roi_gray = gray[y:y + upper_h, x:x + w]
        roi_color = annotated[y:y + upper_h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(15, 15)
        )

        eye_list = []
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # green eye box
            eye_list.append({"x": int(ex), "y": int(ey), "w": int(ew), "h": int(eh)})

        results["faces"].append({
            "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "num_eyes": int(len(eyes)),
            "eyes": eye_list
        })

    return annotated, results


def mean_brightness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


# ---------------------------------------------------------------------------
# Lighting condition simulation
# ---------------------------------------------------------------------------
def adjust_brightness_contrast(img, brightness=0, contrast=0):
    img = img.astype(np.float32)
    if contrast != 0:
        f = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = f * (img - 128) + 128
    img = img + brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def add_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_lighting_conditions(image):
    return {
        "original": image,
        "bright_overexposed": adjust_brightness_contrast(image, brightness=80, contrast=20),
        "dim_low_light": adjust_brightness_contrast(image, brightness=-80, contrast=0),
        "very_dark": adjust_brightness_contrast(image, brightness=-120, contrast=-10),
        "low_contrast_hazy": adjust_brightness_contrast(image, brightness=0, contrast=-90),
        "high_contrast_harsh": adjust_brightness_contrast(image, brightness=0, contrast=80),
        "gamma_dark_indoor": adjust_gamma(image, gamma=0.4),
        "gamma_bright_daylight": adjust_gamma(image, gamma=2.2),
        "noisy_low_light": add_gaussian_noise(adjust_brightness_contrast(image, brightness=-60), sigma=20),
    }


def run_lighting_evaluation(image, output_dir):
    conditions = generate_lighting_conditions(image)
    rows = []

    for name, variant in conditions.items():
        annotated, results = detect_faces_and_eyes(variant)
        brightness = mean_brightness(variant)
        total_eyes = sum(f["num_eyes"] for f in results["faces"])

        cv2.imwrite(os.path.join(output_dir, f"cond_{name}.png"), annotated)

        rows.append({
            "condition": name,
            "mean_brightness": round(brightness, 1),
            "faces_detected": results["num_faces"],
            "total_eyes_detected": total_eyes,
        })
        print(f"[{name:22s}] brightness={brightness:6.1f}  faces={results['num_faces']}  eyes={total_eyes}")

    csv_path = os.path.join(output_dir, "lighting_evaluation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "mean_brightness", "faces_detected", "total_eyes_detected"])
        writer.writeheader()
        writer.writerows(rows)

    # Comparison chart
    cond_names = [r["condition"] for r in rows]
    brightness_vals = [r["mean_brightness"] for r in rows]
    faces_vals = [r["faces_detected"] for r in rows]
    eyes_vals = [r["total_eyes_detected"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = range(len(cond_names))

    axes[0].bar(x, brightness_vals, color="#4C72B0")
    axes[0].set_ylabel("Mean Brightness (0-255)")
    axes[0].set_title("Image Brightness per Lighting Condition")

    width = 0.35
    axes[1].bar([i - width / 2 for i in x], faces_vals, width, label="Faces detected", color="#55A868")
    axes[1].bar([i + width / 2 for i in x], eyes_vals, width, label="Eyes detected", color="#C44E52")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(cond_names, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Detections")
    axes[1].set_title("Detection Accuracy per Lighting Condition")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_chart.png"), dpi=150)
    plt.close(fig)

    print(f"\nLighting evaluation CSV: {csv_path}")
    print(f"Accuracy chart: {os.path.join(output_dir, 'accuracy_chart.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ==========================================================================
    # >>> PUT YOUR IMAGE HERE <<<
    # Put your photo in the SAME FOLDER as this script, then just change the
    # filename below to match it (e.g. "myphoto.jpg", "student1.png", etc.)
    # ==========================================================================
    input_path = "111.jpg"          # <-- CHANGE THIS to your image filename
    output_dir = "results"            # <-- folder where results will be saved
    # ==========================================================================

    # (Optional) still allow running from terminal like:
    #   python3 attendance_face_eye_detection.py myphoto.jpg results
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        print(f"ERROR: could not read image at {input_path}")
        sys.exit(1)

    # 1. Detection on the original image
    annotated, results = detect_faces_and_eyes(image)
    results["mean_brightness"] = mean_brightness(image)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_img_path = os.path.join(output_dir, f"{base}_detected.png")
    out_json_path = os.path.join(output_dir, f"{base}_results.json")

    # Write filename + detection summary as text directly on the image
    info_lines = [
        f"File: {os.path.basename(input_path)}",
        f"Faces: {results['num_faces']}  |  Eyes: {sum(f['num_eyes'] for f in results['faces'])}",
        f"Brightness: {results['mean_brightness']:.1f}",
    ]
    y0 = 25
    for i, line in enumerate(info_lines):
        y = y0 + i * 22
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_img_path, annotated)
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Faces detected: {results['num_faces']}")
    for i, face in enumerate(results["faces"]):
        print(f"  Face {i+1}: eyes detected = {face['num_eyes']}")
    print(f"Mean brightness: {results['mean_brightness']:.1f}")
    print(f"Annotated image: {out_img_path}")
    print(f"JSON results: {out_json_path}\n")

    # 2. Lighting condition evaluation (simulated variants of the same image)
    print("Running lighting condition evaluation...")
    run_lighting_evaluation(image, output_dir)

    # 3. Show the detected image on screen (original photo with boxes drawn)
    show_image_popup(annotated, "Detected Face & Eyes (original photo)")


def show_image_popup(image_bgr, title="Result"):
    """Pops up a window showing the image. Works reliably in VS Code."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(title)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
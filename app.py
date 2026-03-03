import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_utils import classify_gesture, thumb_index_distance

from pycaw.pycaw import AudioUtilities

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume  # BUILT-IN property

min_vol, max_vol = volume.GetVolumeRange()[:2]

# -------------------------
# MediaPipe Hand Landmarker
# -------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

def draw_play_pause(frame, playing):
    if playing:
        # ⏸ Pause icon
        cv2.rectangle(frame, (20, 70), (30, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 70), (50, 110), (0, 255, 0), -1)
    else:
        # ▶ Play icon
        pts = [(20, 70), (20, 110), (55, 90)]
        cv2.fillPoly(frame, [np.array(pts)], (0, 255, 0))

def draw_volume_bar(frame, volume_percent):
    x, y = 20, 130
    w, h = 200, 20

    # Outline
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Filled bar
    fill = int(w * volume_percent)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, 255, 0), -1)

    cv2.putText(
        frame,
        f"Volume: {int(volume_percent * 100)}%",
        (x, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

# -------------------------
# Video Player
# -------------------------
video = cv2.VideoCapture("demo.mp4")
playing = False

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    gesture = "NO HAND"

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        gesture, states = classify_gesture(landmarks)

        # 🎮 Actions
        if gesture == "OPEN_PALM":
            playing = True
            
        elif gesture == "FIST":
            playing = False

        # 🔊 Volume via pinch
        distance = thumb_index_distance(landmarks)
        distance = np.clip(distance, 0.02, 0.15)
        volume_percent = np.interp(distance, [0.02, 0.15], [0.0, 1.0])

        # Apply system volume
        vol = np.interp(volume_percent, [0, 1], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)
        # Draw landmarks
        for lm in landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        draw_play_pause(frame, playing)
        draw_volume_bar(frame, volume_percent)

    cv2.putText(
    frame,
    f"Gesture: {gesture}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
    )

    # 🎥 Video display
    if playing:
        ret_vid, vid_frame = video.read()
        if ret_vid:
            cv2.imshow("Video Player", vid_frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
# Records labeled clips for training the classifier.
# Only landmark numbers are saved. No video or images are stored.
#
# keys:  1-6 pick a behavior | SPACE record a clip | C calibrate | Q quit

import glob
import os
import time

import cv2
import numpy as np

import config
import detector
import features
import storage
import ui
import vision


def count_clips():
    counts = {}
    for label in config.LABELS:
        counts[label] = len(glob.glob(os.path.join(config.DATASET_DIR, label + "__*.csv")))
    return counts


def main():
    perception = vision.Perception()
    cap = vision.open_camera()
    if cap is None:
        print("Could not open a camera. Check camera permission and CAMERA_ORDER in config.py")
        return

    title_font = ui.load_font(22)
    small_font = ui.load_font(15)

    counts = count_clips()
    selected = 0
    phase = "idle"        # idle, countdown or recording
    phase_end = 0
    rows = []
    calibrator = detector.AutoCalibrator()
    baseline = None
    saved_until = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        face_points, hands_points, face_result = perception.process(frame)
        now = time.time()

        calibrator.update(face_points, hands_points)
        baseline = calibrator.baseline

        # contact with calibration, and contact without it (to compare the two in train.py)
        region = None
        raw_region = None
        if face_points is not None and hands_points:
            region = detector.check_touch(face_points, hands_points, baseline)
            raw_region = detector.check_touch(face_points, hands_points, None)

        if phase == "countdown" and now >= phase_end:
            phase = "recording"
            phase_end = now + config.RECORD_SECONDS
            rows = []

        if phase == "recording":
            vec = features.frame_features(face_points, hands_points)
            rows.append([1.0 if region else 0.0, 1.0 if raw_region else 0.0] + list(vec))
            if now >= phase_end:
                label = config.LABELS[selected]
                storage.save_clip(label, rows)
                counts[label] += 1
                phase = "idle"
                saved_until = now + 2

        # ---- drawing ----
        name = config.BEHAVIOR_NAMES[config.LABELS[selected]]
        if phase == "countdown":
            status_text = "Get ready: " + str(int(phase_end - now) + 1)
            mode = "snoozed"
        elif phase == "recording":
            status_text = "Recording: " + name + "  " + str(int(phase_end - now) + 1) + "s"
            mode = "alert"
        else:
            status_text = "Ready: " + name
            mode = "clear"

        ui.draw_hands(frame, hands_points, mode)

        lines = []
        for i, label in enumerate(config.LABELS):
            marker = "> " if i == selected else "   "
            lines.append(marker + str(i + 1) + "  " + config.BEHAVIOR_NAMES[label] +
                         "  (" + str(counts[label]) + " clips)")

        hint_text = "1-6 behavior   SPACE record   C calibrate   Q quit"
        badge_text = "Clip saved" if now < saved_until else ""
        if baseline is None and phase == "idle":
            badge_text = badge_text or "Calibrating"

        frame = ui.draw_ui(frame, status_text, mode, lines, hint_text, badge_text,
                           title_font, small_font)
        cv2.imshow("Collect training data", frame)
        key = cv2.waitKey(1) & 0xFF
        if ord("A") <= key <= ord("Z"):
            key += 32     # treat capital letters (Caps Lock or Shift) like lowercase

        if key == ord("q"):
            break
        elif ord("1") <= key < ord("1") + len(config.LABELS) and phase == "idle":
            selected = key - ord("1")
        elif key == ord(" "):
            if phase == "idle":
                phase = "countdown"
                phase_end = time.time() + config.COUNTDOWN_SECONDS
            else:
                phase = "idle"   # cancel, nothing is saved
                rows = []
        elif key == ord("c") and face_points is not None and hands_points:
            calibrator.set_manual(detector.hand_ratio(hands_points[0], detector.face_width(face_points)))
            print("calibrated by hand, baseline ratio:", round(calibrator.baseline, 2))

    cap.release()
    cv2.destroyAllWindows()
    print("clips per behavior:", count_clips())


if __name__ == "__main__":
    main()
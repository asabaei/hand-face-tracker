import math
from collections import deque

import numpy as np
import mediapipe as mp

import config

mp_face_mesh = mp.solutions.face_mesh

# fingertips + the joint below each fingertip count as touch points
TOUCH_POINTS = [4, 8, 12, 16, 20, 3, 7, 11, 15, 19]


def get_ids(connections):
    ids = set()
    for a, b in connections:
        ids.add(a)
        ids.add(b)
    return ids


# face mesh landmark ids for each region
LIPS = get_ids(mp_face_mesh.FACEMESH_LIPS)
EYES = (
    get_ids(mp_face_mesh.FACEMESH_LEFT_EYE)
    | get_ids(mp_face_mesh.FACEMESH_RIGHT_EYE)
    | get_ids(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    | get_ids(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
)
NOSE = {1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 129, 358, 49, 279, 48, 278, 219, 439}

LIPS_IDX = np.array(sorted(LIPS))
EYES_IDX = np.array(sorted(EYES))
NOSE_IDX = np.array(sorted(NOSE))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_points(landmarks, w, h):
    points = []
    for p in landmarks.landmark:
        points.append((p.x * w, p.y * h))
    return points


def face_width(face_points):
    return dist(face_points[234], face_points[454])


def hand_ratio(hand_points, width):
    # use the bigger of two hand measurements so it still works when the hand is tilted
    size = max(dist(hand_points[0], hand_points[9]), dist(hand_points[5], hand_points[17]))
    return size / width


def get_region(d, point_y, brow_y, width):
    # d = distances from one finger point to every face landmark
    if d[EYES_IDX].min() < config.EYE_RADIUS * width:
        return "eyes"
    if d[NOSE_IDX].min() < config.NOSE_RADIUS * width:
        return "nose"
    if d[LIPS_IDX].min() < config.MOUTH_RADIUS * width:
        return "mouth"
    if point_y < brow_y:
        return "forehead"
    return "cheek_chin"


def check_touch(face_points, hands_points, baseline_ratio):
    # returns the region being touched, or None
    face_array = np.array(face_points)
    width = face_width(face_points)
    brow_y = face_array[EYES_IDX, 1].min()

    best_d = config.NEAR_FRACTION * width
    best_region = None

    for hand_points in hands_points:
        ratio = hand_ratio(hand_points, width)

        # a hand that looks bigger than it does when touching is closer to the camera
        if baseline_ratio is None:
            if ratio > config.MAX_HAND_RATIO:
                continue
        elif ratio > baseline_ratio * config.CALIBRATION_TOLERANCE:
            continue

        for p in TOUCH_POINTS:
            px, py = hand_points[p]
            d = np.hypot(face_array[:, 0] - px, face_array[:, 1] - py)
            if d.min() < best_d:
                best_d = d.min()
                best_region = get_region(d, py, brow_y, width)

    return best_region


def tight_contact_ratio(face_points, hands_points):
    # if a fingertip is almost exactly on the face, return how big that hand looks. else None
    face_array = np.array(face_points)
    width = face_width(face_points)

    best_d = config.AUTOCAL_TIGHT_FRACTION * width
    best_ratio = None
    for hand_points in hands_points:
        for p in TOUCH_POINTS:
            px, py = hand_points[p]
            d = np.hypot(face_array[:, 0] - px, face_array[:, 1] - py).min()
            if d < best_d:
                best_d = d
                best_ratio = hand_ratio(hand_points, width)
    return best_ratio


class AutoCalibrator:
    # learns the "hand size at face depth" baseline from your own touches,
    # so you don't have to press C. Pressing C still works and takes over.

    def __init__(self, enabled=True):
        self.enabled = enabled and config.AUTO_CALIBRATE
        self.samples = deque(maxlen=config.AUTOCAL_MAX_SAMPLES)
        self.streak = 0
        self.baseline = None
        self.locked = False    # True once you set it by hand

    def set_manual(self, value):
        self.baseline = value
        self.locked = True

    def update(self, face_points, hands_points):
        if self.locked or not self.enabled:
            return
        if face_points is None or not hands_points:
            self.streak = 0
            return

        ratio = tight_contact_ratio(face_points, hands_points)
        if ratio is None:
            self.streak = 0
            return

        # once we have a baseline, ignore hands that look much bigger or smaller (hovering)
        if self.baseline is not None:
            if abs(ratio / self.baseline - 1) > config.AUTOCAL_OUTLIER:
                self.streak = 0
                return

        # the contact has to last a moment before we trust it
        self.streak += 1
        if self.streak >= config.AUTOCAL_HOLD_FRAMES:
            self.samples.append(ratio)
            if len(self.samples) >= config.AUTOCAL_MIN_SAMPLES:
                self.baseline = float(np.median(self.samples))
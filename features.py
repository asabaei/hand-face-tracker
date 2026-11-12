# Turns landmarks into numbers a classifier can use.
#   frame_features:  one frame -> a vector
#   window_features: a window of frames -> one vector (stats + motion)

import math

import numpy as np

import config
import detector

FINGERTIPS = [4, 8, 12, 16, 20]
FAR = 3.0   # distance used when no hand is visible (in face widths)

REGION_SETS = [
    ("eyes", detector.EYES_IDX),
    ("nose", detector.NOSE_IDX),
    ("mouth", detector.LIPS_IDX),
    ("any", None),
]

FRAME_FEATURE_NAMES = []
for i in range(21):
    FRAME_FEATURE_NAMES += ["hand" + str(i) + "_x", "hand" + str(i) + "_y"]
for name, idx in REGION_SETS:
    FRAME_FEATURE_NAMES += [name + "_min_dist", name + "_mean_dist"]
FRAME_FEATURE_NAMES += ["hand_ratio", "openness", "spread", "pinch",
                        "angle_sin", "angle_cos", "hand_present"]
N_FRAME = len(FRAME_FEATURE_NAMES)

PRESENT = FRAME_FEATURE_NAMES.index("hand_present")
EYES_MIN = FRAME_FEATURE_NAMES.index("eyes_min_dist")
NOSE_MIN = FRAME_FEATURE_NAMES.index("nose_min_dist")
MOUTH_MIN = FRAME_FEATURE_NAMES.index("mouth_min_dist")
ANY_MIN = FRAME_FEATURE_NAMES.index("any_min_dist")

MOTION_NAMES = ["mean_speed", "max_speed", "flips_x", "flips_y",
                "dwell_eyes", "dwell_nose", "dwell_mouth", "dwell_any"]
WINDOW_FEATURE_NAMES = []
for stat in ["mean", "std", "min", "max"]:
    for name in FRAME_FEATURE_NAMES:
        WINDOW_FEATURE_NAMES.append(name + "_" + stat)
WINDOW_FEATURE_NAMES += MOTION_NAMES


def absent_vector():
    vec = np.zeros(N_FRAME)
    for i, name in enumerate(FRAME_FEATURE_NAMES):
        if name.endswith("_dist"):
            vec[i] = FAR
    return vec


def pick_hand(face_array, hands_points):
    # the hand closest to the face
    best = None
    best_d = 1e9
    for hand in hands_points:
        tips = np.array([hand[i] for i in FINGERTIPS])
        d = np.hypot(tips[:, None, 0] - face_array[None, :, 0],
                     tips[:, None, 1] - face_array[None, :, 1]).min()
        if d < best_d:
            best_d = d
            best = hand
    return best


def frame_features(face_points, hands_points):
    if face_points is None or not hands_points:
        return absent_vector()

    face_array = np.array(face_points)
    width = detector.face_width(face_points)
    origin = face_array[1]   # nose tip

    hand = np.array(pick_hand(face_array, hands_points))
    rel = (hand - origin) / width                     # hand position relative to the face
    tips = hand[FINGERTIPS]

    d = np.hypot(tips[:, None, 0] - face_array[None, :, 0],
                 tips[:, None, 1] - face_array[None, :, 1]) / width   # (5, 468)

    region_feats = []
    for name, idx in REGION_SETS:
        sub = d if idx is None else d[:, idx]
        per_tip = sub.min(axis=1)
        region_feats.append(per_tip.min())
        region_feats.append(per_tip.mean())

    size = max(detector.dist(hand[0], hand[9]), detector.dist(hand[5], hand[17]), 1e-6)
    ratio = size / width
    openness = np.hypot(*(tips - hand[0]).T).mean() / size
    spread = np.std(tips, axis=0).mean() / width
    pinch = detector.dist(tips[0], tips[1]) / width
    angle = math.atan2(hand[9][1] - hand[0][1], hand[9][0] - hand[0][0])

    vec = list(rel.flatten()) + region_feats
    vec += [ratio, openness, spread, pinch, math.sin(angle), math.cos(angle), 1.0]
    return np.array(vec)


def window_features(window):
    # window is an array of shape (frames, N_FRAME)
    window = np.array(window)
    n = len(window)

    stats = np.concatenate([window.mean(axis=0), window.std(axis=0),
                            window.min(axis=0), window.max(axis=0)])

    # motion of the fingertips (only between frames where a hand is visible)
    cols = []
    for t in FINGERTIPS:
        cols += [2 * t, 2 * t + 1]
    tips_xy = window[:, cols].reshape(n, 5, 2)
    present = window[:, PRESENT] > 0.5
    valid = present[1:] & present[:-1]

    vel = np.diff(tips_xy, axis=0)
    speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
    speed = speed[valid]
    mean_speed = speed.mean() if len(speed) else 0.0
    max_speed = speed.max() if len(speed) else 0.0

    # how many times the hand changes direction (rubbing / biting move back and forth)
    centroid_vel = np.diff(tips_xy.mean(axis=1), axis=0)[valid]
    flips = []
    for axis in (0, 1):
        v = centroid_vel[:, axis] if len(centroid_vel) else np.array([])
        v = v[np.abs(v) > 0.003]
        s = np.sign(v)
        flips.append(float((np.diff(s) != 0).sum()) if len(s) > 1 else 0.0)

    dwell = []
    for idx in [EYES_MIN, NOSE_MIN, MOUTH_MIN, ANY_MIN]:
        dwell.append(float((window[:, idx] < config.NEAR_FRACTION).mean()))

    motion = [mean_speed, max_speed, flips[0], flips[1]] + dwell
    return np.concatenate([stats, np.array(motion)])
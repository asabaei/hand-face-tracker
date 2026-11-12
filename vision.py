import cv2
import mediapipe as mp

import config
import detector

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


def open_camera():
    for index in config.CAMERA_ORDER:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            return cap
    return None


class Perception:
    # runs the hand and face models on a frame and returns pixel coordinates

    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.6)
        self.face = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6)
        self.face_points = None
        self.face_lost = 0

    def process(self, frame):
        # frame is a BGR image. returns face_points, hands_points, face_result
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_result = self.hands.process(rgb)
        face_result = self.face.process(rgb)

        # keep the last face position for a few frames if a hand blocks the face
        if face_result.multi_face_landmarks:
            self.face_points = detector.get_points(face_result.multi_face_landmarks[0], w, h)
            self.face_lost = 0
        else:
            self.face_lost += 1
            if self.face_lost > config.FACE_LOST_FRAMES:
                self.face_points = None

        hands_points = []
        if hand_result.multi_hand_landmarks:
            for hand in hand_result.multi_hand_landmarks:
                hands_points.append(detector.get_points(hand, w, h))

        return self.face_points, hands_points, face_result
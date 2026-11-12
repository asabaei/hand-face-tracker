import argparse
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

import alerts
import config
import detector
import diagnose
import features
import routine
import storage
import ui
import vision

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

FEEDBACK_KEYS = {}
for key_name in config.FEEDBACK_KEYS:
    FEEDBACK_KEYS[ord(key_name)] = config.FEEDBACK_KEYS[key_name]


class TouchState:
    # turns per-frame "is there a touch" into start/end events.
    # timing is in seconds, so it behaves the same at any frame rate

    def __init__(self):
        self.reset()

    def reset(self):
        self.run_start = None     # when the current stretch of contact began
        self.last_contact = 0
        self.in_event = False
        self.start = 0
        self.region = None

    def update(self, region, now):
        # returns None, ("start", region, start, 0) or ("end", region, start, duration)
        if region:
            if self.run_start is None:
                self.run_start = now
            self.last_contact = now
        elif self.run_start is not None and now - self.last_contact > config.GRACE_SECONDS:
            return self.finish()

        if region and not self.in_event and now - self.run_start >= config.PERSISTENCE_SECONDS:
            self.in_event = True
            self.start = self.run_start
            self.region = region
            return ("start", region, self.start, 0)
        return None

    def finish(self):
        # ends the current event right now (also used for snooze and quitting)
        result = None
        if self.in_event:
            result = ("end", self.region, self.start, max(self.last_contact - self.start, 0.0))
        self.reset()
        return result


class EventInfo:
    # what the classifier thought during the current touch

    def __init__(self):
        self.reset()

    def reset(self):
        self.alerted = False
        self.habit = None
        self.votes = Counter()

    def behavior(self):
        if self.habit:
            return self.habit
        if self.votes:
            return self.votes.most_common(1)[0][0]
        return "unclassified"


def fmt_time(seconds):
    seconds = max(int(seconds), 0)
    return str(seconds // 60) + ":" + str(seconds % 60).zfill(2)


def record_end(result, info, session_events, event_log):
    kind, region, start, duration = result
    storage.log_event(start, region, info.behavior(), info.alerted, duration)
    session_events.append((region, duration))
    event_log.append((start, start + duration, info.alerted))


def habit_indices(classes):
    return [i for i, c in enumerate(classes) if c in config.HABIT_LABELS]


def run_retrain(retrain):
    # runs in the background so the camera doesn't freeze
    try:
        import train
        retrain["result"] = train.quick_train()
    except Exception as error:
        print("retrain failed:", error)
        retrain["result"] = (None, "Retrain failed")
    retrain["running"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_DIR, help="folder for the log files")
    parser.add_argument("--model", nargs="?", const=config.MODEL_PATH, default=None,
                        help="use the trained behavior classifier (default file: model.joblib)")
    parser.add_argument("--no-autocal", action="store_true",
                        help="turn off automatic calibration (press C to calibrate by hand)")
    parser.add_argument("--no-mouth-rule", action="store_true",
                        help="let the classifier call nail biting anywhere on the face (the original behavior)")
    parser.add_argument("--routine", action="store_true",
                        help="show the scripted test routine and record its marked touches automatically")
    args = parser.parse_args()
    storage.setup(args.data)

    # optional classifier (you can also create one while running with the T key)
    model = None
    classes = []
    habit_idx = []
    use_model = False
    if args.model:
        try:
            model = joblib.load(args.model)["model"]
        except FileNotFoundError:
            print("No model at " + args.model + ". Run collect.py and train.py first.")
            return
        classes = list(model.classes_)
        habit_idx = habit_indices(classes)
        use_model = True

    perception = vision.Perception()
    cap = vision.open_camera()
    if cap is None:
        print("Could not open a camera. Check camera permission and CAMERA_ORDER in config.py")
        return

    title_font = ui.load_font(22)
    small_font = ui.load_font(15)
    face_style = mp_draw.DrawingSpec(color=(130, 130, 130), thickness=1)

    state = TouchState()
    info = EventInfo()
    manager = alerts.AlertManager()
    calibrator = detector.AutoCalibrator(enabled=not args.no_autocal)
    baseline = None
    show_face = False
    snooze_until = 0
    label_start = None
    labels_count = 0

    # classifier and feedback state
    window = deque(maxlen=config.WINDOW_FRAMES)
    feedback = deque(maxlen=config.FEEDBACK_FRAMES)
    retrain = {"running": False, "result": None}
    frame_count = 0
    pred_label = None
    pred_prob = 0
    confirm_count = 0
    habit_confirmed = False
    current_habit = None
    region_window = deque(maxlen=config.WINDOW_FRAMES)   # where the hand touched over the last ~second
    alert_text = ""
    alert_until = 0         # keeps the last alert on screen for a moment (display only)
    use_mouth_rule = config.NAIL_BITING_NEEDS_MOUTH and not args.no_mouth_rule
    corrections = 0
    message = ""
    message_until = 0

    session_start = time.time()
    session_events = []     # list of (region, duration)
    event_log = []          # list of (start, end, alerted)
    frame_log = []          # list of (time, hand_seen, contact_without_size_check, contact)
    label_intervals = []    # marked touches

    # scripted routine (only with --routine)
    steps = routine.build() if args.routine else []
    total_actions = sum(1 for step in steps if not step.get("rest"))
    routine_started = False
    routine_done = False
    routine_end = None
    step_index = 0
    step_start = 0
    step_end = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        face_points, hands_points, face_result = perception.process(frame)
        now = time.time()

        calibrator.update(face_points, hands_points)
        baseline = calibrator.baseline

        # ---- scripted routine: move to the next step when its time is up ----
        if routine_started and not routine_done and now >= step_end:
            step = steps[step_index]
            if step["marked"]:
                storage.log_label(step_start, step_end)
                label_intervals.append((step_start, step_end))
                labels_count += 1
            step_index += 1
            if step_index < len(steps):
                step_start = step_end
                step_end = step_start + steps[step_index]["seconds"]
            else:
                routine_done = True
                routine_end = step_end
                print("routine finished, press Q")

        # a retrain finished in the background: switch to the new model
        if retrain["result"] is not None:
            bundle, text = retrain["result"]
            retrain["result"] = None
            if bundle is not None:
                model = bundle["model"]
                classes = list(model.classes_)
                habit_idx = habit_indices(classes)
                pred_label = None
                confirm_count = 0
                habit_confirmed = False
            message = text
            message_until = now + 4

        vec = features.frame_features(face_points, hands_points)
        window.append(vec)
        frame_count += 1

        # ---- classifier: look at the last second of movement ----
        if model is not None and len(window) == config.WINDOW_FRAMES \
                and frame_count % config.CLASSIFY_EVERY == 0:
            x = features.window_features(np.array(window)).reshape(1, -1)
            probs = model.predict_proba(x)[0]

            # nail biting needs fingers at the mouth. if during the last second the hand touched
            # the face but never at the mouth (or just under the nose), it can't be nail biting
            touched = any(r is not None for r in region_window)
            at_mouth = any(r in ("mouth", "nose") for r in region_window)
            if use_mouth_rule and touched and not at_mouth and "nail_biting" in classes:
                probs = probs.copy()
                probs[classes.index("nail_biting")] = 0.0
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            best = int(probs.argmax())
            pred_label = classes[best]
            pred_prob = probs[best]

            p_habit = sum(probs[i] for i in habit_idx)
            current_habit = classes[max(habit_idx, key=lambda i: probs[i])]
            if p_habit >= config.CLASSIFIER_THRESHOLD:
                confirm_count += 1
            else:
                confirm_count = 0
            habit_confirmed = confirm_count >= config.CLASSIFIER_CONFIRM

            if state.in_event:
                info.votes[pred_label] += 1

        # ---- touch events ----
        gating = model is not None and use_model   # only alert when the classifier agrees
        snoozed = now < snooze_until
        region = None
        raw_region = None

        # in routine mode, nothing counts before the routine starts or after it ends
        routine_paused = args.routine and (not routine_started or routine_done)

        if snoozed or routine_paused:
            state.reset()
            info.reset()
        else:
            if face_points is not None and hands_points:
                region = detector.check_touch(face_points, hands_points, baseline)
                raw_region = detector.check_touch(face_points, hands_points, None)

            result = state.update(region, now)
            if result and result[0] == "start":
                info.reset()
                if not gating:
                    manager.fire(result[1], now)    # no classifier: alert right away
                    info.alerted = True
                    alert_text = "Touching: " + config.REGION_NAMES[result[1]]
                    alert_until = now + config.ALERT_DISPLAY_SECONDS
            elif result and result[0] == "end":
                record_end(result, info, session_events, event_log)

            if gating and state.in_event and not info.alerted and habit_confirmed:
                manager.fire(state.region, now)
                info.alerted = True
                info.habit = current_habit
                alert_text = config.BEHAVIOR_NAMES[current_habit] + ": " + config.REGION_NAMES[state.region]
                alert_until = now + config.ALERT_DISPLAY_SECONDS

        region_window.append(region)
        frame_log.append((now, 1 if hands_points else 0, 1 if raw_region else 0, 1 if region else 0))

        # keep the last few seconds so a correction key can save them
        feedback.append([1.0 if region else 0.0, 1.0 if raw_region else 0.0] + list(vec))

        # ---- drawing ----
        if show_face and face_result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_result.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_style,
            )

        alerting = state.in_event and info.alerted
        if snoozed:
            mode = "snoozed"
            status_text = "Snoozed " + fmt_time(snooze_until - now)
        elif alerting:
            mode = "alert"
            if info.habit:
                status_text = config.BEHAVIOR_NAMES[info.habit] + ": " + config.REGION_NAMES[state.region]
            else:
                status_text = "Touching: " + config.REGION_NAMES[state.region]
        elif now < alert_until:
            mode = "alert"          # the touch just ended, keep the alert readable for a moment
            status_text = alert_text
        elif state.in_event and gating:
            # contact detected, but the classifier hasn't called it a habit
            mode = "clear"
            if pred_label is not None and pred_label not in config.HABIT_LABELS:
                status_text = config.BEHAVIOR_NAMES[pred_label] + ", no alert"
            else:
                status_text = "Checking: " + config.REGION_NAMES[state.region]
        elif state.in_event:
            mode = "clear"
            status_text = "Contact: " + config.REGION_NAMES[state.region]
        else:
            mode = "clear"
            status_text = "No contact"

        ui.draw_hands(frame, hands_points, mode)

        lines = ["Alerts this session: " + str(manager.count)]
        if labels_count > 0 or label_start is not None:
            lines.append("Marked touches: " + str(labels_count))
        if corrections > 0:
            lines.append("Corrections saved: " + str(corrections))
        if model is not None and pred_label is not None:
            prefix = "Behavior: " if gating else "Model (watching): "
            percent = str(int(pred_prob * 100))
            lines.append(prefix + config.BEHAVIOR_NAMES[pred_label] + " (" + percent + "%)")
        if face_points is not None and hands_points:
            width = detector.face_width(face_points)
            line = "Hand scale " + str(round(detector.hand_ratio(hands_points[0], width), 2))
            if baseline is not None:
                kind = "manual" if calibrator.locked else "auto"
                line += "  |  baseline " + str(round(baseline, 2)) + " (" + kind + ")"
            lines.append(line)

        if baseline is None and calibrator.enabled:
            hint_text = "Touch your face for a few seconds to calibrate (or press C)"
        elif baseline is None:
            hint_text = "Touch your face, then press C to calibrate"
        else:
            hint_text = "X wrong alert   B P E real habit   T retrain   M model   Q quit"

        if args.routine:
            if not routine_started:
                if baseline is None and calibrator.enabled:
                    hint_text = "Touch your face until this changes, then press SPACE"
                else:
                    hint_text = "Press SPACE to start the routine"
            elif routine_done:
                hint_text = "Routine finished. Press Q"
            else:
                step = steps[step_index]
                left = str(int(step_end - now) + 1)
                if step.get("rest"):
                    upcoming = steps[step_index + 1]["name"] if step_index + 1 < len(steps) else "done"
                    hint_text = "Rest, hands in lap (" + left + ")   next: " + upcoming
                else:
                    hint_text = "NOW: " + step["name"] + "  (" + left + ")"

        if retrain["running"]:
            badge_text = "Training..."
        elif now < message_until:
            badge_text = message
        elif label_start is not None:
            badge_text = "Marking real touch"
        elif args.routine and routine_started and not routine_done:
            number = sum(1 for step in steps[:step_index + 1] if not step.get("rest"))
            badge_text = "Item " + str(number) + " of " + str(total_actions)
        else:
            badge_text = ""

        frame = ui.draw_ui(frame, status_text, mode, lines, hint_text, badge_text,
                           title_font, small_font)

        cv2.imshow("Hand-Face Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if ord("A") <= key <= ord("Z"):
            key += 32     # treat capital letters (Caps Lock or Shift) like lowercase

        if key == ord("q"):
            break
        elif key == ord("f"):
            show_face = not show_face
        elif key == ord("c") and face_points is not None and hands_points:
            calibrator.set_manual(detector.hand_ratio(hands_points[0], detector.face_width(face_points)))
            baseline = calibrator.baseline
            print("calibrated by hand, baseline ratio:", round(baseline, 2))
        elif key == ord("s"):
            if snoozed:
                snooze_until = 0
            else:
                result = state.finish()
                if result:
                    record_end(result, info, session_events, event_log)
                snooze_until = time.time() + config.SNOOZE_MINUTES * 60
        elif key == ord(" ") and args.routine and not routine_started:
            routine_started = True
            step_index = 0
            step_start = time.time()
            step_end = step_start + steps[0]["seconds"]
            session_start = step_start
            print("routine started")
        elif key == ord("g") and not args.routine:
            if label_start is None:
                label_start = time.time()
                print("marking started")
            else:
                storage.log_label(label_start, time.time())
                label_intervals.append((label_start, time.time()))
                label_start = None
                labels_count += 1
                print("marked touch #" + str(labels_count) + " saved")
        elif key in FEEDBACK_KEYS:
            # tell it what the last few seconds really were
            label = FEEDBACK_KEYS[key]
            if len(feedback) >= 2 * config.WINDOW_FRAMES:
                storage.save_clip(label, list(feedback), "fb")
                corrections += 1
                message = "Saved as: " + config.BEHAVIOR_NAMES[label]
            else:
                message = "Not enough footage yet"
            message_until = time.time() + 2
        elif key == ord("t"):
            if retrain["running"]:
                message = "Already training"
                message_until = time.time() + 2
            else:
                retrain["running"] = True
                threading.Thread(target=run_retrain, args=(retrain,), daemon=True).start()
        elif key == ord("m"):
            if model is None:
                message = "No model yet. Save corrections, then press T"
            else:
                use_model = not use_model
                message = "Model gates alerts" if use_model else "Model watching only"
            message_until = time.time() + 2

    # wrap up
    result = state.finish()
    if result:
        record_end(result, info, session_events, event_log)
    if label_start is not None:
        storage.log_label(label_start, time.time())
        label_intervals.append((label_start, time.time()))
        labels_count += 1

    print("marked touches this session:", labels_count)
    session_end = routine_end if routine_end else time.time()
    storage.log_session(session_start, session_end, session_events)
    print(storage.summary_text(session_start, session_end, session_events))
    diagnose.report(frame_log, label_intervals, event_log, model is not None and use_model, args.data)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()